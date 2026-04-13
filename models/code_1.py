import pandas as pd
import numpy as np
import requests
import io
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

try:
    import xg_engine
    XG_AVAILABLE = True
except ImportError:
    XG_AVAILABLE = False
    print("Trinity: xg_engine not found — running without xG features")

# =============================================================================
# TRINITY ENGINE  (God Mode — Multi-League)
# =============================================================================

class GodModeEngine:
    def __init__(self, league_code='E0'):
        self.league_code = league_code
        self.SEASONS = ['1516','1617','1718','1819','1920','2021','2122','2223','2324','2425','2526']
        self.ODDS_URL = f"https://www.football-data.co.uk/mmz4281/{{}}/{self.league_code}.csv"

        self.master_df    = None
        self.model        = None
        self.scaler       = None
        self.curr_elo_dict = {}
        self.current_teams = []

        # xG state (populated after load_xg_data)
        self.xg_features   = pd.DataFrame()
        self.xg_loaded     = False

        # Feature list — extended with xG columns when available
        self.base_features = [
            'Elo_Diff',
            'EMA_SOT_Diff',
            'EMA_Corn_Diff',
            'Eff_Trend_Diff',
            'Home_EMA_SOT',        # NEW: split home/away form
            'Away_EMA_SOT',
        ]
        self.xg_features_list = [
            'Diff_xg_diff',        # net xG dominance diff (home - away)
            'Diff_xg_ratio',       # xG ratio diff
            'H_xg_against_ema',    # how many xG the home team typically concedes
            'A_xg_against_ema',    # how many xG the away team typically concedes
        ]
        self.features = self.base_features  # overwritten after xG load

    # -------------------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------------------
    def load_data(self):
        dfs = []
        for s in self.SEASONS:
            try:
                url = self.ODDS_URL.format(s)
                r   = requests.get(url, timeout=15)
                if r.status_code != 200:
                    continue
                df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
                df = df.dropna(how='all')

                if s in ['2425', '2526']:
                    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()
                    self.current_teams = sorted(teams)

                cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','HS','AS','HST','AST','HC','AC']
                df   = df[[c for c in cols if c in df.columns]]
                dfs.append(df)
            except Exception as e:
                print(f"Trinity load_data: season {s} failed — {e}")

        if not dfs:
            return False

        df = pd.concat(dfs, ignore_index=True)
        col_map = {
            'Date': 'date', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team',
            'FTHG': 'home_goals', 'FTAG': 'away_goals',
            'HST':  'home_shots_on_target', 'AST': 'away_shots_on_target',
            'HC':   'home_corners', 'AC': 'away_corners',
        }
        df.rename(columns=col_map, inplace=True)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

        for c in ['home_shots_on_target','away_shots_on_target','home_corners','away_corners']:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].mean())
            else:
                df[c] = 0.0

        self.master_df = df
        return True

    def load_xg_data(self):
        """Fetch Understat xG and attach to the engine. Safe to call — won't crash on failure."""
        if not XG_AVAILABLE:
            return
        try:
            raw_xg = xg_engine.fetch_xg_data(self.league_code)
            if raw_xg.empty:
                print(f"Trinity: No xG data for {self.league_code} — continuing without it")
                return
            self.xg_features = xg_engine.build_rolling_xg_features(raw_xg, ewm_span=6)
            self.xg_loaded   = True
            print(f"Trinity: xG features loaded for {self.league_code}.")
        except Exception as e:
            print(f"Trinity: xG load failed ({e}) — continuing without xG")

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    def engineer_features(self):
        df = self.master_df.copy()

        # 1. ELO (with margin-of-victory scaling)
        curr_elo = {t: 1500.0 for t in pd.concat([df['home_team'], df['away_team']]).unique()}
        df['home_elo'] = 1500.0
        df['away_elo'] = 1500.0
        k = 20

        for i, row in df.iterrows():
            h, a = row['home_team'], row['away_team']
            h_elo, a_elo = curr_elo.get(h, 1500.0), curr_elo.get(a, 1500.0)
            df.at[i, 'home_elo'] = h_elo
            df.at[i, 'away_elo'] = a_elo

            gd  = abs(row['home_goals'] - row['away_goals'])
            mov = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)

            if row['home_goals'] > row['away_goals']:   res = 1.0
            elif row['home_goals'] == row['away_goals']: res = 0.5
            else:                                        res = 0.0

            e_h = 1 / (1 + 10 ** (-(h_elo - a_elo) / 400))
            curr_elo[h] += k * mov * (res - e_h)
            curr_elo[a] += k * mov * ((1 - res) - (1 - e_h))

        self.curr_elo_dict = curr_elo

        # 2. EMA form — SPLIT by home/away context (fixes the blending bug)
        def make_stream(df, is_home):
            if is_home:
                s = df[['date','home_team','home_goals','home_shots_on_target','home_corners']].copy()
                s.columns = ['date','team','goals','sot','corners']
                s['is_home'] = 1
            else:
                s = df[['date','away_team','away_goals','away_shots_on_target','away_corners']].copy()
                s.columns = ['date','team','goals','sot','corners']
                s['is_home'] = 0
            return s

        home_stream = make_stream(df, True)
        away_stream = make_stream(df, False)
        full_stream = pd.concat([home_stream, away_stream]).sort_values(['team','date'])

        # All-context EMA (blended) — kept for Eff_Trend_Diff
        ema_all = full_stream.groupby('team')[['goals','sot','corners']].transform(
            lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
        )
        full_stream[['ema_goals','ema_sot','ema_corners']] = ema_all

        # Home-specific EMA for SOT
        home_only = home_stream.sort_values(['team','date'])
        home_only['home_ema_sot'] = home_only.groupby('team')['sot'].transform(
            lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
        )

        # Away-specific EMA for SOT
        away_only = away_stream.sort_values(['team','date'])
        away_only['away_ema_sot'] = away_only.groupby('team')['sot'].transform(
            lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
        )

        # Merge all-context EMA back
        df = df.merge(
            full_stream[full_stream['is_home']==1][['date','team','ema_goals','ema_sot','ema_corners']],
            left_on=['date','home_team'], right_on=['date','team'], how='left'
        ).rename(columns={'ema_goals':'h_ema_goals','ema_sot':'h_ema_sot','ema_corners':'h_ema_corn'}).drop(columns=['team'])

        df = df.merge(
            full_stream[full_stream['is_home']==0][['date','team','ema_goals','ema_sot','ema_corners']],
            left_on=['date','away_team'], right_on=['date','team'], how='left'
        ).rename(columns={'ema_goals':'a_ema_goals','ema_sot':'a_ema_sot','ema_corners':'a_ema_corn'}).drop(columns=['team'])

        # Merge home-specific SOT EMA
        df = df.merge(
            home_only[['date','team','home_ema_sot']],
            left_on=['date','home_team'], right_on=['date','team'], how='left'
        ).drop(columns=['team'])

        # Merge away-specific SOT EMA
        df = df.merge(
            away_only[['date','team','away_ema_sot']],
            left_on=['date','away_team'], right_on=['date','team'], how='left'
        ).drop(columns=['team'])

        # 3. Base feature differentials
        df['Elo_Diff']      = df['home_elo'] - df['away_elo']
        df['EMA_SOT_Diff']  = df['h_ema_sot'] - df['a_ema_sot']
        df['EMA_Corn_Diff'] = df['h_ema_corn'] - df['a_ema_corn']

        h_eff = df['h_ema_goals'] / (df['h_ema_sot'] + 0.1)
        a_eff = df['a_ema_goals'] / (df['a_ema_sot'] + 0.1)
        df['Eff_Trend_Diff'] = h_eff - a_eff

        # Context-split SOT (new features)
        avg_sot = df['h_ema_sot'].mean()
        df['Home_EMA_SOT'] = df['home_ema_sot'].fillna(avg_sot)
        df['Away_EMA_SOT'] = df['away_ema_sot'].fillna(avg_sot)

        # 4. Merge xG features if available
        if self.xg_loaded and not self.xg_features.empty:
            df = xg_engine.merge_xg_into_match_df(df, self.xg_features)
            self.features = self.base_features + self.xg_features_list
            print(f"Trinity: Using {len(self.features)} features (with xG).")
        else:
            # Zero-fill xG columns so dropna doesn't remove rows
            for col in self.xg_features_list:
                df[col] = 0.0
            self.features = self.base_features
            print(f"Trinity: Using {len(self.features)} features (no xG).")

        # 5. Target
        conditions = [df['home_goals'] > df['away_goals'], df['home_goals'] == df['away_goals']]
        df['target'] = np.select(conditions, [2, 1], default=0)

        self.master_df = df.dropna(subset=self.base_features).copy()

    # -------------------------------------------------------------------------
    # TRAINING  (with TimeSeriesSplit validation + calibration)
    # -------------------------------------------------------------------------
    def train_trinity_model(self):
        df = self.master_df
        if df.empty:
            return

        X = df[self.features]
        y = df['target']

        self.scaler  = StandardScaler()
        X_scaled     = self.scaler.fit_transform(X)

        # Recency weights
        weights = np.exp(np.linspace(0, 4, len(X)))

        # --- TimeSeriesSplit validation (honest OOS evaluation) ---
        tscv = TimeSeriesSplit(n_splits=5)
        fold_losses = []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_tr        = weights[train_idx]

            # Quick XGB for validation only
            val_xgb = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                objective='multi:softprob', num_class=3,
                random_state=42, eval_metric='mlogloss', verbosity=0
            )
            val_xgb.fit(X_tr, y_tr, sample_weight=w_tr)
            proba = val_xgb.predict_proba(X_val)
            fold_losses.append(log_loss(y_val, proba))

        mean_ll = np.mean(fold_losses)
        std_ll  = np.std(fold_losses)
        print(f"Trinity [{self.league_code}] CV log-loss: {mean_ll:.4f} (±{std_ll:.4f})")

        # Store for /the-ai display (4.1)
        self.cv_metrics = {
            'log_loss': round(float(mean_ll), 4),
            'log_loss_std': round(float(std_ll), 4),
        }

        # --- Train full ensemble on all data ---
        lr  = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
        rf  = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        # FIX: was 'multi:softmax' — changed to 'multi:softprob' for proper probabilities
        xgb_mod = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            objective='multi:softprob', num_class=3,
            random_state=42, eval_metric='mlogloss', verbosity=0
        )

        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_mod)],
            voting='soft', weights=[1, 1, 3]
        )
        ensemble.fit(X_scaled, y, sample_weight=weights)

        # Calibrate probabilities using isotonic regression
        # (ensures 65% confidence actually wins ~65% of the time)
        self.model = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
        self.model.fit(X_scaled, y)

        print(f"Trinity [{self.league_code}] training complete.")

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------
    def predict_match(self, h_team, a_team):
        df_h = self.master_df[
            (self.master_df['home_team'] == h_team) | (self.master_df['away_team'] == h_team)
        ]
        df_a = self.master_df[
            (self.master_df['home_team'] == a_team) | (self.master_df['away_team'] == a_team)
        ]

        if df_h.empty or df_a.empty:
            return None

        row_h = df_h.iloc[-1]
        row_a = df_a.iloc[-1]

        def get_stat(row, team, stat):
            return row[f'h_{stat}'] if row['home_team'] == team else row[f'a_{stat}']

        h_elo = self.curr_elo_dict.get(h_team, 1500.0)
        a_elo = self.curr_elo_dict.get(a_team, 1500.0)

        # Base features
        elo_diff   = h_elo - a_elo
        sot_diff   = get_stat(row_h, h_team, 'ema_sot') - get_stat(row_a, a_team, 'ema_sot')
        corn_diff  = get_stat(row_h, h_team, 'ema_corn') - get_stat(row_a, a_team, 'ema_corn')
        h_eff      = get_stat(row_h, h_team, 'ema_goals') / (get_stat(row_h, h_team, 'ema_sot') + 0.1)
        a_eff      = get_stat(row_a, a_team, 'ema_goals') / (get_stat(row_a, a_team, 'ema_sot') + 0.1)
        eff_diff   = h_eff - a_eff

        # Context-split SOT
        avg_sot       = self.master_df['h_ema_sot'].mean()
        home_ema_sot  = row_h.get('Home_EMA_SOT', avg_sot)
        away_ema_sot  = row_a.get('Away_EMA_SOT', avg_sot)

        feature_vals = {
            'Elo_Diff':      elo_diff,
            'EMA_SOT_Diff':  sot_diff,
            'EMA_Corn_Diff': corn_diff,
            'Eff_Trend_Diff': eff_diff,
            'Home_EMA_SOT':  home_ema_sot,
            'Away_EMA_SOT':  away_ema_sot,
        }

        # xG features
        if self.xg_loaded and not self.xg_features.empty:
            h_xg = xg_engine.get_current_xg_stats(h_team, self.xg_features)
            a_xg = xg_engine.get_current_xg_stats(a_team, self.xg_features)
            feature_vals['Diff_xg_diff']    = h_xg['xg_diff_ema']  - a_xg['xg_diff_ema']
            feature_vals['Diff_xg_ratio']   = h_xg['xg_ratio_ema'] - a_xg['xg_ratio_ema']
            feature_vals['H_xg_against_ema'] = h_xg['xg_against_ema']
            feature_vals['A_xg_against_ema'] = a_xg['xg_against_ema']
        else:
            for col in self.xg_features_list:
                feature_vals[col] = 0.0

        input_vec   = pd.DataFrame([feature_vals], columns=self.features)
        input_scaled = self.scaler.transform(input_vec)

        probs = self.model.predict_proba(input_scaled)[0]
        # probs order from CalibratedClassifierCV matches original class order: 0=Away, 1=Draw, 2=Home
        return {
            'A': probs[0], 'D': probs[1], 'H': probs[2],
            'H_Elo': int(h_elo), 'A_Elo': int(a_elo)
        }


# =============================================================================
# INTERFACE BLOCK (Multi-League)
# =============================================================================

_engines = {}


def get_model_1_prediction(home_team, away_team, league_code='E0'):
    global _engines

    if league_code not in _engines:
        print(f"Initializing Trinity Engine for {league_code}...")
        engine = GodModeEngine(league_code=league_code)

        if not engine.load_data():
            print(f"Trinity: Failed to load data for {league_code}")
            return None

        # Load xG data (non-fatal if it fails)
        engine.load_xg_data()

        engine.engineer_features()
        engine.train_trinity_model()
        _engines[league_code] = engine
        print(f"Trinity [{league_code}] ready.")

    try:
        engine = _engines[league_code]
        pred   = engine.predict_match(home_team, away_team)
        if pred is None:
            return None
        return {
            'home_prob': pred['H'],
            'draw_prob': pred['D'],
            'away_prob': pred['A'],
        }
    except Exception as e:
        print(f"Trinity runtime error ({league_code}): {e}")
        return None
