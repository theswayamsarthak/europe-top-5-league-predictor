import logging
log = logging.getLogger(__name__)
import pandas as pd
import numpy as np
import requests
import io
import warnings
import optuna
from scipy.stats import poisson
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import xg_engine
    XG_AVAILABLE = True
except ImportError:
    XG_AVAILABLE = False
    print("Anchor/Rebel: xg_engine not found — running without xG features")

try:
    import injuries_engine
    INJURIES_AVAILABLE = True
except ImportError:
    INJURIES_AVAILABLE = False
    print("Anchor/Rebel: injuries_engine not found — running without injury features")


# =============================================================================
# 1. ELO ENGINE
# =============================================================================
class EloTracker:
    def __init__(self, k_factor=20, base_rating=1500, home_adv=75):
        self.ratings  = {}
        self.k        = k_factor
        self.base     = base_rating
        self.home_adv = home_adv

    def get_rating(self, team):
        return self.ratings.get(team, self.base)

    def update(self, home, away, result, goal_diff):
        r_home = self.get_rating(home)
        r_away = self.get_rating(away)
        e_home = 1 / (1 + 10 ** ((r_away - (r_home + self.home_adv)) / 400))
        e_away = 1 - e_home

        if result == 'H':   s_home, s_away = 1, 0
        elif result == 'A': s_home, s_away = 0, 1
        else:               s_home, s_away = 0.5, 0.5

        # Margin-of-victory multiplier
        if goal_diff <= 1:   mult = 1.0
        elif goal_diff == 2: mult = 1.5
        else:                mult = (11 + goal_diff) / 8

        k = self.k * mult
        self.ratings[home] = r_home + k * (s_home - e_home)
        self.ratings[away] = r_away + k * (s_away - e_away)


# =============================================================================
# 2. HYBRID PIPELINE  (Anchor + Rebel twin-engine)
# =============================================================================
class HybridPipeline:
    def __init__(self, league_code='E0'):
        self.league_code = league_code
        self.config = {
            'seasons':    ['1617','1718','1819','1920','2021','2122','2223','2324','2425','2526'],
            'base_url':   f"https://www.football-data.co.uk/mmz4281/{{}}/{self.league_code}.csv",
            'elo_k':      20,
            'elo_home_adv': 75,
            'ewm_span':   6,
            'split_ratio': 0.85,
            'decay_alpha': 0.90,
            'xgb_params': {
                'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.025,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'objective': 'multi:softprob', 'num_class': 3,
                'n_jobs': -1, 'random_state': 42, 'verbosity': 0,
            }
        }

        self.data      = None
        self.feat_data = None
        self.proc_data = None
        self.target_map = {'A': 0, 'D': 1, 'H': 2}

        # Stored final ELO state — populated after feature_engineering
        # Avoids replaying all history on every prediction call
        self.final_elo = None

        # xG state
        self.xg_features_df = pd.DataFrame()
        self.xg_loaded      = False

        # Feature sets (populated in feature_engineering)
        self.features_anchor = []
        self.features_rebel  = []

        # Models & scalers
        self.model_anchor  = None
        self.model_rebel   = None
        self.scaler_anchor = None
        self.scaler_rebel  = None
        self.thresh_anchor = 0.30
        self.thresh_rebel  = 0.30

    # -------------------------------------------------------------------------
    # A. DATA INGESTION
    # -------------------------------------------------------------------------
    def fetch_data(self):
        print(f"[{self.league_code}] Fetching historical data...")
        frames  = []
        headers = {'User-Agent': 'Mozilla/5.0'}
        req_cols = [
            'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR',
            'HS','AS','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR',
            'B365H','B365D','B365A'
        ]

        for i, season in enumerate(self.config['seasons']):
            try:
                r = requests.get(self.config['base_url'].format(season), headers=headers, timeout=15)
                if r.status_code != 200:
                    continue
                df = pd.read_csv(io.StringIO(r.content.decode('utf-8')), on_bad_lines='skip')
                df['Date']      = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                df['Season_ID'] = i
                for c in req_cols:
                    if c not in df.columns:
                        df[c] = np.nan
                frames.append(df[req_cols + ['Season_ID']])
            except Exception as e:
                print(f"  [{self.league_code}] Season {season} skipped: {e}")
                continue

        if not frames:
            print(f"[{self.league_code}] No data loaded.")
            return False

        self.data = (
            pd.concat(frames, ignore_index=True)
            .sort_values('Date')
            .reset_index(drop=True)
        )
        self.data = self.data.dropna(subset=['HomeTeam','AwayTeam','FTR'])
        self.data['Target'] = self.data['FTR'].map(self.target_map)
        print(f"[{self.league_code}] Loaded {len(self.data)} matches.")
        return True

    # -------------------------------------------------------------------------
    # B. FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    def _calc_implied_odds(self, df):
        df['Imp_H'] = 1 / df['B365H']
        df['Imp_D'] = 1 / df['B365D']
        df['Imp_A'] = 1 / df['B365A']
        df[['Imp_H','Imp_D','Imp_A']] = df[['Imp_H','Imp_D','Imp_A']].fillna(0.33)
        return df

    def _calc_odds_poisson(self, df):
        """Odds-derived Poisson — used only by Anchor (market-aware model)."""
        def row_poisson(row):
            ph = 1/row['B365H'] if row['B365H'] > 0 else 0.33
            pa = 1/row['B365A'] if row['B365A'] > 0 else 0.33
            mu_h = max(0.1, 1.35 + (ph - pa) * 1.5)
            mu_a = max(0.1, 1.15 + (pa - ph) * 1.0)
            p_h = p_d = p_a = 0.0
            for h in range(6):
                for a in range(6):
                    p = poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a)
                    if h > a:    p_h += p
                    elif h == a: p_d += p
                    else:        p_a += p
            return pd.Series([p_h, p_d, p_a])

        probs = df.apply(row_poisson, axis=1)
        probs.columns = ['Math_Prob_H','Math_Prob_D','Math_Prob_A']
        return pd.concat([df, probs], axis=1)

    def _calc_stats_poisson(self, df):
        """
        Stats-derived Poisson — uses rolling goals-for as attack lambda.
        This is independent of odds and therefore valid for the Rebel model.
        Computed AFTER rolling stats so H_Roll_Adj_GF is available.
        """
        def row_poisson(row):
            mu_h = max(0.3, row.get('H_Roll_Adj_GF', 1.35))
            mu_a = max(0.3, row.get('A_Roll_Adj_GF', 1.15))
            p_h = p_d = p_a = 0.0
            for h in range(7):
                for a in range(7):
                    p = poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a)
                    if h > a:    p_h += p
                    elif h == a: p_d += p
                    else:        p_a += p
            return pd.Series([p_h, p_d, p_a])

        probs = df.apply(row_poisson, axis=1)
        probs.columns = ['Stats_Prob_H','Stats_Prob_D','Stats_Prob_A']
        return pd.concat([df, probs], axis=1)

    def _calc_rolling_stats(self, df):
        home_games = df[['Date','HomeTeam','FTHG','FTAG','HS','AS','HST','AST',
                         'HC','AC','HF','AF','HY','AY','HR','AR','FTR']].copy()
        home_games.columns = ['Date','Team','GF','GA','SF','SA','STF','STA',
                               'CF','CA','Fouls','FoulsAg','Yel','YelAg','Red','RedAg','Res']
        home_games['IsHome']   = 1
        home_games['Opponent'] = df['AwayTeam'].values

        away_games = df[['Date','AwayTeam','FTAG','FTHG','AS','HS','AST','HST',
                         'AC','HC','AF','HF','AY','HY','AR','HR','FTR']].copy()
        away_games.columns = ['Date','Team','GF','GA','SF','SA','STF','STA',
                               'CF','CA','Fouls','FoulsAg','Yel','YelAg','Red','RedAg','Res']
        away_games['IsHome']   = 0
        away_games['Opponent'] = df['HomeTeam'].values

        all_games = pd.concat([home_games, away_games]).sort_values('Date').reset_index(drop=True)

        all_games['Win']    = (
            ((all_games['IsHome']==1) & (all_games['Res']=='H')) |
            ((all_games['IsHome']==0) & (all_games['Res']=='A'))
        ).astype(int)
        all_games['Draw']   = (all_games['Res']=='D').astype(int)
        all_games['Points'] = all_games['Win']*3 + all_games['Draw']
        all_games['DiscPoints']  = all_games['Yel'] + (all_games['Red'] * 10)
        all_games['TotalGoals']  = all_games['GF'] + all_games['GA']

        # Opponent-adjusted goals
        avg_ga = all_games['GA'].mean()
        all_games['Raw_Roll_GA'] = all_games.groupby('Team')['GA'].transform(
            lambda x: x.shift(1).ewm(span=10).mean().fillna(avg_ga)
        )
        lookup = (
            all_games[['Date','Team','Raw_Roll_GA']]
            .rename(columns={'Team':'Opponent','Raw_Roll_GA':'Opp_Def_Strength'})
        )
        all_games = pd.merge(all_games, lookup, on=['Date','Opponent'], how='left')
        all_games['Opp_Def_Strength'] = all_games['Opp_Def_Strength'].fillna(avg_ga)
        all_games['Adj_GF'] = (all_games['GF'] / (all_games['Opp_Def_Strength'] + 0.1)) * avg_ga
        all_games['Adj_SF'] = (all_games['SF'] / (all_games['Opp_Def_Strength']*5 + 0.1)) * (avg_ga * 5)

        cols_to_roll = ['Adj_GF','GA','Adj_SF','SA','STF','STA','CF','CA',
                        'Points','Fouls','DiscPoints','TotalGoals']
        defaults     = all_games[cols_to_roll].mean().to_dict()

        grouped_mean = all_games.groupby('Team')[cols_to_roll].transform(
            lambda x: x.shift(1).ewm(span=self.config['ewm_span']).mean()
        ).fillna(value=defaults)
        grouped_mean.columns = [f'Roll_{c}' for c in cols_to_roll]

        grouped_std = all_games.groupby('Team')[['GF']].transform(
            lambda x: x.shift(1).rolling(self.config['ewm_span']).std()
        ).fillna(all_games['GF'].std())
        grouped_std.columns = ['Roll_GF_Std']

        all_games['Specific_Form'] = all_games.groupby(['Team','IsHome'])['Points'].transform(
            lambda x: x.shift(1).ewm(span=5).mean()
        ).fillna(1.3)

        all_games = pd.concat([all_games, grouped_mean, grouped_std], axis=1)
        feat_cols = list(grouped_mean.columns) + ['Roll_GF_Std','Specific_Form']

        h_stats = (
            all_games[all_games['IsHome']==1][['Date','Team'] + feat_cols]
            .rename(columns={c: f'H_{c}' for c in feat_cols})
        )
        a_stats = (
            all_games[all_games['IsHome']==0][['Date','Team'] + feat_cols]
            .rename(columns={c: f'A_{c}' for c in feat_cols})
        )

        df = pd.merge(df, h_stats, left_on=['Date','HomeTeam'], right_on=['Date','Team'], how='left').drop(columns=['Team'])
        df = pd.merge(df, a_stats, left_on=['Date','AwayTeam'], right_on=['Date','Team'], how='left').drop(columns=['Team'])
        return df.dropna()

    def _calc_elo(self, df):
        """ELO with margin-of-victory multiplier. Stores final ratings on self.final_elo."""
        elo          = EloTracker(k_factor=self.config['elo_k'], home_adv=self.config['elo_home_adv'])
        feature_rows = []
        df           = df.sort_values('Date').reset_index(drop=True)

        for _, row in df.iterrows():
            h, a, res = row['HomeTeam'], row['AwayTeam'], row['FTR']
            h_elo     = elo.get_rating(h)
            a_elo     = elo.get_rating(a)
            gd        = abs(row['FTHG'] - row['FTAG'])
            elo.update(h, a, res, gd)
            feature_rows.append({'H_Elo': h_elo, 'A_Elo': a_elo})

        # Store final ELO state for prediction time (no more full rebuild per call)
        self.final_elo = elo

        feat_df = pd.DataFrame(feature_rows, index=df.index)
        return pd.concat([df, feat_df], axis=1)

    def feature_engineering(self):
        print(f"[{self.league_code}] Engineering features...")
        df = self.data.copy()

        df = self._calc_implied_odds(df)
        df = self._calc_odds_poisson(df)
        df = self._calc_rolling_stats(df)
        df = self._calc_elo(df)

        # Derived differentials
        df['Diff_Elo']         = df['H_Elo'] - df['A_Elo']
        df['Abs_Diff_Elo']     = df['Diff_Elo'].abs()
        df['Diff_ShotDom']     = (df['H_Roll_Adj_SF']/(df['H_Roll_Adj_SF']+df['H_Roll_SA']+0.1)) - \
                                  (df['A_Roll_Adj_SF']/(df['A_Roll_Adj_SF']+df['A_Roll_SA']+0.1))
        df['Diff_SOTDom']      = (df['H_Roll_STF']/(df['H_Roll_STF']+df['H_Roll_STA']+0.1)) - \
                                  (df['A_Roll_STF']/(df['A_Roll_STF']+df['A_Roll_STA']+0.1))
        df['Diff_CornDom']     = (df['H_Roll_CF']/(df['H_Roll_CF']+df['H_Roll_CA']+0.1)) - \
                                  (df['A_Roll_CF']/(df['A_Roll_CF']+df['A_Roll_CA']+0.1))
        df['Diff_SpecificForm']= df['H_Specific_Form'] - df['A_Specific_Form']
        df['Diff_Volatility']  = df['H_Roll_GF_Std']   - df['A_Roll_GF_Std']
        df['Diff_Aggression']  = df['H_Roll_Fouls']     - df['A_Roll_Fouls']
        df['Diff_Discipline']  = df['H_Roll_DiscPoints']- df['A_Roll_DiscPoints']
        df['Boredom_Score']    = (df['H_Roll_TotalGoals'] + df['A_Roll_TotalGoals']) / 2
        df['Market_Elo_Div']   = df['Imp_H'] - (1 / (1 + 10**((-df['Diff_Elo']-75)/400)))

        # Stats-based Poisson (requires rolling stats to be computed first)
        df = self._calc_stats_poisson(df)

        # --- xG from Understat ---
        if XG_AVAILABLE:
            try:
                raw_xg = xg_engine.fetch_xg_data(self.league_code)
                if not raw_xg.empty:
                    self.xg_features_df = xg_engine.build_rolling_xg_features(raw_xg, ewm_span=6)
                    self.xg_loaded      = True
                    df = xg_engine.merge_xg_into_match_df(df, self.xg_features_df)
                    print(f"[{self.league_code}] xG features merged.")
                else:
                    print(f"[{self.league_code}] Understat returned no data — continuing without xG.")
            except Exception as e:
                print(f"[{self.league_code}] xG load failed: {e} — continuing without xG.")

        # Zero-fill xG columns so feature lists stay consistent
        for col in ['Diff_xg_diff','Diff_xg_ratio','H_xg_against_ema','A_xg_against_ema']:
            if col not in df.columns:
                df[col] = 0.0

        # FIX 3.5: Zero-fill injury/lineup features at training time.
        # Historical match data has no injury/lineup info — these features
        # are only populated at live prediction time via injuries_engine.
        # Using 0.0 / 1.0 (neutral) means training treats all historical
        # games as fully-fit full-strength, which is the correct baseline.
        for col in ['Injury_Penalty_H', 'Injury_Penalty_A']:
            df[col] = 0.0    # no penalty = everyone available
        for col in ['Lineup_Str_H', 'Lineup_Str_A']:
            df[col] = 1.0    # full strength = baseline

        # --- FEATURE SET DEFINITIONS ---
        # Rebel: pure performance — no market odds, includes independent Poisson + xG
        # FIX 3.5: Injury_Penalty and Lineup_Str added as pre-match adjustment features.
        # These are zero-filled during training (historical data has no lineup info)
        # and populated at prediction time from injuries_engine.
        self.features_rebel = [
            'Diff_Elo','Diff_ShotDom','Diff_SOTDom','Diff_CornDom',
            'Diff_SpecificForm','Diff_Volatility','Diff_Aggression','Diff_Discipline',
            'Abs_Diff_Elo','Boredom_Score','H_Elo','A_Elo',
            'Diff_xg_diff','Diff_xg_ratio','H_xg_against_ema','A_xg_against_ema',
            'Stats_Prob_H','Stats_Prob_D','Stats_Prob_A',
            'Injury_Penalty_H','Injury_Penalty_A',   # 3.5: weighted missing players
            'Lineup_Str_H','Lineup_Str_A',            # 3.5: confirmed XI strength
        ]

        # Anchor: adds market signals on top of Rebel
        self.features_anchor = self.features_rebel + [
            'Market_Elo_Div','Imp_H','Imp_D','Imp_A',
            'Math_Prob_H','Math_Prob_D','Math_Prob_A',
        ]

        # Filter to only cols that actually exist (safety guard)
        self.features_rebel  = [c for c in self.features_rebel  if c in df.columns]
        self.features_anchor = [c for c in self.features_anchor if c in df.columns]

        self.feat_data = df.copy()
        train_cols     = list(set(self.features_anchor + ['Target','Season_ID','Date']))
        self.proc_data = df[[c for c in train_cols if c in df.columns]].copy()

        print(f"[{self.league_code}] Ready — Rebel ({len(self.features_rebel)} feats) | "
              f"Anchor ({len(self.features_anchor)} feats)")

    # -------------------------------------------------------------------------
    # C. TRAINING
    # -------------------------------------------------------------------------
    def calculate_decay_weights(self, season_ids):
        max_season = season_ids.max()
        return self.config['decay_alpha'] ** (max_season - season_ids.values)

    def tune_hyperparameters(self, features, y, season_ids, n_trials=10):
        print(f"   Tuning hyperparameters ({n_trials} trials)...")

        def objective(trial):
            param = {
                'n_estimators':      trial.suggest_int('n_estimators', 200, 600),
                'max_depth':         trial.suggest_int('max_depth', 3, 6),
                'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample':         trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'objective': 'multi:softprob', 'num_class': 3,
                'n_jobs': -1, 'random_state': 42, 'verbosity': 0,
            }
            tscv   = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(features):
                X_tr, X_val = features.iloc[train_idx], features.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                w_tr        = self.calculate_decay_weights(season_ids.iloc[train_idx])
                scaler      = RobustScaler()
                X_tr_s      = scaler.fit_transform(X_tr)
                X_val_s     = scaler.transform(X_val)
                model       = XGBClassifier(**param)
                model.fit(X_tr_s, y_tr, sample_weight=w_tr)
                proba       = model.predict_proba(X_val_s)
                scores.append(log_loss(y_val, proba))
            return np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        print(f"   Best log-loss: {study.best_value:.4f} | Params: {study.best_params}")
        return {**self.config['xgb_params'], **study.best_params}

    def train_models(self, perform_tuning=False):
        print(f"\n[{self.league_code}] Training Anchor + Rebel...")

        y          = self.proc_data['Target']
        season_ids = self.proc_data['Season_ID']
        split      = int(len(self.proc_data) * self.config['split_ratio'])
        w_train    = self.calculate_decay_weights(season_ids.iloc[:split])
        y_train    = y.iloc[:split]
        y_test     = y.iloc[split:]

        def train_engine(name, feature_cols):
            print(f"   Training {name}...")
            avail_cols = [c for c in feature_cols if c in self.proc_data.columns]
            X          = self.proc_data[avail_cols]
            X_train    = X.iloc[:split]
            X_test     = X.iloc[split:]

            params = (
                self.tune_hyperparameters(X_train, y_train, season_ids.iloc[:split])
                if perform_tuning else self.config['xgb_params']
            )

            scaler      = RobustScaler()
            X_train_s   = scaler.fit_transform(X_train)
            X_test_s    = scaler.transform(X_test)

            model = XGBClassifier(**params)
            model.fit(X_train_s, y_train, sample_weight=w_train)

            # Calibrate probabilities
            cal_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            cal_model.fit(X_train_s, y_train)

            # Evaluate on holdout
            y_proba    = cal_model.predict_proba(X_test_s)
            best_f1    = 0
            best_thresh = 0.30
            for thresh in np.arange(0.25, 0.50, 0.01):
                preds = []
                for row in y_proba:
                    if row[1] > thresh: preds.append(1)
                    else: preds.append(2 if row[2] > row[0] else 0)
                score = f1_score(y_test, preds, average='macro')
                if score > best_f1:
                    best_f1    = score
                    best_thresh = thresh

            oos_ll = log_loss(y_test, y_proba)
            print(f"   {name} — thresh: {best_thresh:.2f} | F1: {best_f1:.4f} | log-loss: {oos_ll:.4f}")
            return cal_model, scaler, best_thresh, avail_cols, round(float(best_f1), 4), round(float(oos_ll), 4)

        (self.model_anchor, self.scaler_anchor, self.thresh_anchor,
         self.features_anchor, anchor_f1, anchor_ll) = train_engine("ANCHOR", self.features_anchor)
        (self.model_rebel,  self.scaler_rebel,  self.thresh_rebel,
         self.features_rebel,  rebel_f1,  rebel_ll)  = train_engine("REBEL",  self.features_rebel)

        # Store training metrics for /the-ai display (4.1)
        self.train_metrics = {
            'anchor': {'f1': anchor_f1, 'log_loss': anchor_ll},
            'rebel':  {'f1': rebel_f1,  'log_loss': rebel_ll},
        }

    # -------------------------------------------------------------------------
    # D. HISTORY (for dashboard Recent Results table)
    # -------------------------------------------------------------------------
    def get_history(self, n=10):
        if self.proc_data is None:
            return []

        last_games = self.feat_data.tail(n).copy()
        history    = []

        for _, row in last_games.iterrows():
            def predict_row(feature_cols, model, scaler):
                fvec = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
                return model.predict_proba(scaler.transform(fvec))[0]

            probs_reb = predict_row(self.features_rebel,  self.model_rebel,  self.scaler_rebel)
            probs_anc = predict_row(self.features_anchor, self.model_anchor, self.scaler_anchor)

            def lbl(idx): return 'HOME' if idx == 2 else ('AWAY' if idx == 0 else 'DRAW')

            history.append({
                'home_team':   row['HomeTeam'],
                'away_team':   row['AwayTeam'],
                'date':        row['Date'].strftime('%d-%m'),
                'fixture':     f"{row['HomeTeam']} vs {row['AwayTeam']}",
                'result':      f"{row['FTHG']:.0f}-{row['FTAG']:.0f} ({row['FTR']})",
                'pred_rebel':  lbl(int(np.argmax(probs_reb))),
                'pred_anchor': lbl(int(np.argmax(probs_anc))),
            })

        return history[::-1]


# =============================================================================
# INTERFACE BLOCK
# =============================================================================
_pipelines = {}


def reset_pipeline():
    global _pipelines
    print("Twin-Engine reset: clearing all league memory...")
    _pipelines = {}


def get_history_data(league_code='E0'):
    if league_code not in _pipelines:
        return []
    return _pipelines[league_code].get_history(10)


def get_model_metrics(league_code='E0'):
    """Returns stored training metrics for Anchor and Rebel (F1 + log-loss)."""
    if league_code not in _pipelines:
        return {}
    return getattr(_pipelines[league_code], 'train_metrics', {})


def get_model_2_prediction(home_team, away_team, home_odds, draw_odds, away_odds, league_code='E0'):
    global _pipelines

    # Initialise pipeline for league if not ready
    if league_code not in _pipelines:
        print(f"Initialising Anchor/Rebel for {league_code}...")
        pipeline = HybridPipeline(league_code=league_code)
        if not pipeline.fetch_data():
            print(f"Anchor/Rebel: data fetch failed for {league_code}")
            return None
        pipeline.feature_engineering()
        pipeline.train_models(perform_tuning=False)
        _pipelines[league_code] = pipeline
        print(f"Anchor/Rebel [{league_code}] ready.")

    ai = _pipelines[league_code]

    try:
        # 1. Get rolling stats from most recent game record for each team
        def get_team_stats(team):
            rows = ai.feat_data[
                (ai.feat_data['HomeTeam'] == team) | (ai.feat_data['AwayTeam'] == team)
            ]
            if rows.empty:
                return None
            last = ai.feat_data.loc[rows.last_valid_index()]
            pfx  = 'H_' if last['HomeTeam'] == team else 'A_'
            return {
                'Adj_SF':   float(last.get(f'{pfx}Roll_Adj_SF',   5.0)),
                'SA':       float(last.get(f'{pfx}Roll_SA',        5.0)),
                'STF':      float(last.get(f'{pfx}Roll_STF',       3.0)),
                'STA':      float(last.get(f'{pfx}Roll_STA',       3.0)),
                'CF':       float(last.get(f'{pfx}Roll_CF',        4.0)),
                'CA':       float(last.get(f'{pfx}Roll_CA',        4.0)),
                'SpecForm': float(last.get(f'{pfx}Specific_Form',  1.0)),
                'GF_Std':   float(last.get(f'{pfx}Roll_GF_Std',   0.5)),
                'Fouls':    float(last.get(f'{pfx}Roll_Fouls',    10.0)),
                'Disc':     float(last.get(f'{pfx}Roll_DiscPoints',1.0)),
                'TotG':     float(last.get(f'{pfx}Roll_TotalGoals',2.5)),
                'Adj_GF':   float(last.get(f'{pfx}Roll_Adj_GF',   1.35)),
            }

        h_s = get_team_stats(home_team)
        a_s = get_team_stats(away_team)
        if not h_s or not a_s:
            return None

        # 2. ELO from stored final state — O(1), no more full replay
        h_elo = ai.final_elo.get_rating(home_team)
        a_elo = ai.final_elo.get_rating(away_team)

        # 3. Stats-based Poisson (Rebel — independent of odds)
        mu_h = max(0.3, h_s['Adj_GF'])
        mu_a = max(0.3, a_s['Adj_GF'])
        stats_p_h = stats_p_d = stats_p_a = 0.0
        for h in range(7):
            for a in range(7):
                p = poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a)
                if h > a:    stats_p_h += p
                elif h == a: stats_p_d += p
                else:        stats_p_a += p

        # 4. Odds-based Poisson (Anchor only)
        ph_mkt = 1 / home_odds
        pa_mkt = 1 / away_odds
        mu_h_m = max(0.1, 1.35 + (ph_mkt - pa_mkt) * 1.5)
        mu_a_m = max(0.1, 1.15 + (pa_mkt - ph_mkt) * 1.0)
        mkt_p_h = mkt_p_d = mkt_p_a = 0.0
        for h in range(6):
            for a in range(6):
                p = poisson.pmf(h, mu_h_m) * poisson.pmf(a, mu_a_m)
                if h > a:    mkt_p_h += p
                elif h == a: mkt_p_d += p
                else:        mkt_p_a += p

        # 5. xG features from Understat
        if ai.xg_loaded and not ai.xg_features_df.empty:
            h_xg = xg_engine.get_current_xg_stats(home_team, ai.xg_features_df)
            a_xg = xg_engine.get_current_xg_stats(away_team, ai.xg_features_df)
        else:
            h_xg = a_xg = {
                'xg_diff_ema': 0.0, 'xg_ratio_ema': 0.5,
                'xg_against_ema': 1.35,
            }

        # 5b. FIX 3.5: Injury + lineup features from API-Football
        # Fetched fresh per prediction — cached for 3 hours inside injuries_engine
        if INJURIES_AVAILABLE:
            try:
                prematch = injuries_engine.get_prematch_features(
                    home_team, away_team, league_code
                )
            except Exception as _e:
                log.warning(f"injuries_engine failed: {_e}")
                prematch = {
                    'Injury_Penalty_H': 0.0, 'Injury_Penalty_A': 0.0,
                    'Lineup_Str_H': 1.0,     'Lineup_Str_A': 1.0,
                }
        else:
            prematch = {
                'Injury_Penalty_H': 0.0, 'Injury_Penalty_A': 0.0,
                'Lineup_Str_H': 1.0,     'Lineup_Str_A': 1.0,
            }

        # 6. Assemble feature vector
        f = {}
        f['Diff_Elo']          = h_elo - a_elo
        f['Abs_Diff_Elo']      = abs(f['Diff_Elo'])
        f['Diff_ShotDom']      = (h_s['Adj_SF']/(h_s['Adj_SF']+h_s['SA']+0.1)) - \
                                  (a_s['Adj_SF']/(a_s['Adj_SF']+a_s['SA']+0.1))
        f['Diff_SOTDom']       = (h_s['STF']/(h_s['STF']+h_s['STA']+0.1)) - \
                                  (a_s['STF']/(a_s['STF']+a_s['STA']+0.1))
        f['Diff_CornDom']      = (h_s['CF']/(h_s['CF']+h_s['CA']+0.1)) - \
                                  (a_s['CF']/(a_s['CF']+a_s['CA']+0.1))
        f['Diff_SpecificForm'] = h_s['SpecForm'] - a_s['SpecForm']
        f['Diff_Volatility']   = h_s['GF_Std']   - a_s['GF_Std']
        f['Diff_Aggression']   = h_s['Fouls']    - a_s['Fouls']
        f['Diff_Discipline']   = h_s['Disc']     - a_s['Disc']
        f['Boredom_Score']     = (h_s['TotG'] + a_s['TotG']) / 2
        f['H_Elo']             = h_elo
        f['A_Elo']             = a_elo
        # Stats Poisson
        f['Stats_Prob_H']      = stats_p_h
        f['Stats_Prob_D']      = stats_p_d
        f['Stats_Prob_A']      = stats_p_a
        # xG
        f['Diff_xg_diff']      = h_xg['xg_diff_ema']  - a_xg['xg_diff_ema']
        f['Diff_xg_ratio']     = h_xg['xg_ratio_ema'] - a_xg['xg_ratio_ema']
        f['H_xg_against_ema']  = h_xg['xg_against_ema']
        f['A_xg_against_ema']  = a_xg['xg_against_ema']
        # FIX 3.5: Injury + lineup adjustment features
        f['Injury_Penalty_H'] = prematch['Injury_Penalty_H']
        f['Injury_Penalty_A'] = prematch['Injury_Penalty_A']
        f['Lineup_Str_H']     = prematch['Lineup_Str_H']
        f['Lineup_Str_A']     = prematch['Lineup_Str_A']

        # Anchor-only market features
        f['Market_Elo_Div']    = (1/home_odds) - (1/(1 + 10**((-f['Diff_Elo']-75)/400)))
        f['Imp_H']             = 1 / home_odds
        f['Imp_D']             = 1 / draw_odds
        f['Imp_A']             = 1 / away_odds
        f['Math_Prob_H']       = mkt_p_h
        f['Math_Prob_D']       = mkt_p_d
        f['Math_Prob_A']       = mkt_p_a

        # 7. Predict
        row_rebel  = pd.DataFrame([f]).reindex(columns=ai.features_rebel,  fill_value=0)
        row_anchor = pd.DataFrame([f]).reindex(columns=ai.features_anchor, fill_value=0)

        probs_rebel  = ai.model_rebel.predict_proba(ai.scaler_rebel.transform(row_rebel))[0]
        probs_anchor = ai.model_anchor.predict_proba(ai.scaler_anchor.transform(row_anchor))[0]

        return {
            'anchor': {'home': float(probs_anchor[2]), 'draw': float(probs_anchor[1]), 'away': float(probs_anchor[0])},
            'rebel':  {'home': float(probs_rebel[2]),  'draw': float(probs_rebel[1]),  'away': float(probs_rebel[0])},
        }

    except Exception as e:
        print(f"Anchor/Rebel error ({league_code}): {e}")
        import traceback; traceback.print_exc()
        return None
