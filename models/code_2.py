import pandas as pd
import numpy as np
import requests
import io
import warnings
import optuna
from scipy.stats import poisson
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

# Suppress warnings & Optuna logs
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 1. ADVANCED ELO ENGINE (UNCHANGED)
# =============================================================================
class EloTracker:
    def __init__(self, k_factor=20, base_rating=1500, home_adv=75):
        self.ratings = {}
        self.k = k_factor
        self.base = base_rating
        self.home_adv = home_adv

    def get_rating(self, team):
        return self.ratings.get(team, self.base)

    def update(self, home, away, result, goal_diff):
        r_home = self.get_rating(home)
        r_away = self.get_rating(away)
        e_home = 1 / (1 + 10 ** ((r_away - (r_home + self.home_adv)) / 400))
        e_away = 1 - e_home

        if result == 'H': s_home, s_away = 1, 0
        elif result == 'A': s_home, s_away = 0, 1
        else: s_home, s_away = 0.5, 0.5

        # Dynamic K based on Margin of Victory
        if goal_diff <= 1: mult = 1.0
        elif goal_diff == 2: mult = 1.5
        else: mult = (11 + goal_diff) / 8

        current_k = self.k * mult
        self.ratings[home] = r_home + current_k * (s_home - e_home)
        self.ratings[away] = r_away + current_k * (s_away - e_away)

# =============================================================================
# 2. IMPROVED HYBRID PIPELINE (GENERALIZED FOR ALL LEAGUES)
# =============================================================================
class HybridPipeline:
    def __init__(self, league_code='E0'):
        self.league_code = league_code
        # Centralized Configuration
        self.config = {
            'seasons': ['1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526'],
            # Dynamic URL insertion
            'base_url': f"https://www.football-data.co.uk/mmz4281/{{}}/{self.league_code}.csv",
            'elo_k': 20,
            'elo_home_adv': 75,
            'ewm_span': 6,          # Rolling form span
            'split_ratio': 0.85,    # Train/Test split
            'decay_alpha': 0.90,    # Season decay weight
            'h2h_decay': 0.85,      # Decay factor for H2H matches
            'xgb_params': {         # Default Params
                'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.025,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'objective': 'multi:softprob', 'num_class': 3, 'n_jobs': -1, 'random_state': 42
            }
        }

        self.data = None
        self.feat_data = None # Stores full data with rolling cols
        self.proc_data = None
        self.target_map = {'A': 0, 'D': 1, 'H': 2}

        # Feature Sets
        self.features_anchor = []
        self.features_rebel = []

        # Models & Scalers
        self.model_anchor = None
        self.model_rebel = None
        self.scaler_anchor = None
        self.scaler_rebel = None
        self.thresh_anchor = 0.30
        self.thresh_rebel = 0.30

    # -------------------------------------------------------------------------
    # A. DATA INGESTION
    # -------------------------------------------------------------------------
    def fetch_data(self):
        print(f"📥 [{self.league_code}] Fetching raw data (including current 25/26 season)...")
        frames = []
        headers = {'User-Agent': 'Mozilla/5.0'}
        req_cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR',
                    'HS','AS','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR',
                    'B365H','B365D','B365A']

        for i, season in enumerate(self.config['seasons']):
            try:
                # Format URL with season (league_code is already in base_url)
                r = requests.get(self.config['base_url'].format(season), headers=headers)
                if r.status_code == 200:
                    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')), on_bad_lines='skip')
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                    df['Season_ID'] = i
                    for c in req_cols:
                        if c not in df.columns: df[c] = np.nan
                    frames.append(df[req_cols + ['Season_ID']])
            except: continue

        if not frames:
            print(f"❌ No data found for {self.league_code}")
            return False

        self.data = pd.concat(frames, ignore_index=True).sort_values('Date').reset_index(drop=True)
        self.data = self.data.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'])
        self.data['Target'] = self.data['FTR'].map(self.target_map)
        print(f"✅ [{self.league_code}] Loaded {len(self.data)} matches.")
        return True

    # -------------------------------------------------------------------------
    # B. MODULAR FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    def _calc_implied_odds(self, df):
        """Calculates market implied probabilities."""
        df['Imp_H'] = 1 / df['B365H']
        df['Imp_D'] = 1 / df['B365D']
        df['Imp_A'] = 1 / df['B365A']
        df[['Imp_H','Imp_D','Imp_A']] = df[['Imp_H','Imp_D','Imp_A']].fillna(0.33)
        return df

    def _calc_poisson_math(self, df):
        """Calculates theoretical probabilities using Poisson distribution."""
        def row_poisson(row):
            prob_h = 1/row['B365H'] if row['B365H'] > 0 else 0.33
            prob_a = 1/row['B365A'] if row['B365A'] > 0 else 0.33
            mu_h = max(0.1, 1.35 + (prob_h - prob_a) * 1.5)
            mu_a = max(0.1, 1.15 + (prob_a - prob_h) * 1.0)

            p_h, p_d, p_a = 0, 0, 0
            for h in range(6):
                for a in range(6):
                    p = poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a)
                    if h > a: p_h += p
                    elif h == a: p_d += p
                    else: p_a += p
            return pd.Series([p_h, p_d, p_a])

        poisson_probs = df.apply(row_poisson, axis=1)
        poisson_probs.columns = ['Math_Prob_H', 'Math_Prob_D', 'Math_Prob_A']
        return pd.concat([df, poisson_probs], axis=1)

    def _calc_rolling_stats(self, df):
        """Calculates advanced rolling statistics with league-average fillna."""
        # 1. Transform to Long Format
        home_games = df[['Date','HomeTeam','FTHG','FTAG','HS','AS','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR','FTR']].copy()
        home_games.columns = ['Date','Team','GF','GA','SF','SA','STF','STA','CF','CA','Fouls','FoulsAg','Yel','YelAg','Red','RedAg','Res']
        home_games['IsHome'] = 1
        home_games['Opponent'] = df['AwayTeam']

        away_games = df[['Date','AwayTeam','FTAG','FTHG','AS','HS','AST','HST','AC','HC','AF','HF','AY','HY','AR','HR','FTR']].copy()
        away_games.columns = ['Date','Team','GF','GA','SF','SA','STF','STA','CF','CA','Fouls','FoulsAg','Yel','YelAg','Red','RedAg','Res']
        away_games['IsHome'] = 0
        away_games['Opponent'] = df['HomeTeam']

        all_games = pd.concat([home_games, away_games]).sort_values('Date').reset_index(drop=True)

        # 2. Base Metrics
        all_games['Win'] = (((all_games['IsHome']==1) & (all_games['Res']=='H')) | ((all_games['IsHome']==0) & (all_games['Res']=='A'))).astype(int)
        all_games['Draw'] = (all_games['Res']=='D').astype(int)
        all_games['Points'] = all_games['Win']*3 + all_games['Draw']
        all_games['DiscPoints'] = all_games['Yel'] + (all_games['Red'] * 10)
        all_games['TotalGoals'] = all_games['GF'] + all_games['GA']

        # 3. Opponent Adjustment
        avg_ga = all_games['GA'].mean()
        all_games['Raw_Roll_GA'] = all_games.groupby('Team')['GA'].transform(lambda x: x.shift(1).ewm(span=10).mean().fillna(avg_ga))

        lookup = all_games[['Date', 'Team', 'Raw_Roll_GA']].rename(columns={'Team': 'Opponent', 'Raw_Roll_GA': 'Opp_Def_Strength'})
        all_games = pd.merge(all_games, lookup, on=['Date', 'Opponent'], how='left')
        all_games['Opp_Def_Strength'] = all_games['Opp_Def_Strength'].fillna(avg_ga)

        all_games['Adj_GF'] = (all_games['GF'] / (all_games['Opp_Def_Strength'] + 0.1)) * avg_ga
        all_games['Adj_SF'] = (all_games['SF'] / (all_games['Opp_Def_Strength']*5 + 0.1)) * (avg_ga * 5)

        # 4. Rolling Aggregations
        cols_to_roll = ['Adj_GF', 'GA', 'Adj_SF', 'SA', 'STF', 'STA', 'CF', 'CA', 'Points', 'Fouls', 'DiscPoints', 'TotalGoals']

        defaults = all_games[cols_to_roll].mean().to_dict()

        grouped_mean = all_games.groupby('Team')[cols_to_roll].transform(
            lambda x: x.shift(1).ewm(span=self.config['ewm_span']).mean()
        )
        grouped_mean = grouped_mean.fillna(value=defaults)
        grouped_mean.columns = [f'Roll_{c}' for c in cols_to_roll]

        # 5. Volatility
        grouped_std = all_games.groupby('Team')[['GF']].transform(lambda x: x.shift(1).rolling(self.config['ewm_span']).std())
        grouped_std = grouped_std.fillna(all_games['GF'].std())
        grouped_std.columns = ['Roll_GF_Std']

        # 6. Specific Form
        all_games['Specific_Form'] = all_games.groupby(['Team', 'IsHome'])['Points'].transform(
            lambda x: x.shift(1).ewm(span=5).mean()
        ).fillna(1.3)

        # 7. Merge back
        all_games = pd.concat([all_games, grouped_mean, grouped_std], axis=1)
        feature_cols = list(grouped_mean.columns) + ['Roll_GF_Std', 'Specific_Form']

        h_stats = all_games[all_games['IsHome']==1][['Date','Team'] + feature_cols].rename(columns={c: f'H_{c}' for c in feature_cols})
        a_stats = all_games[all_games['IsHome']==0][['Date','Team'] + feature_cols].rename(columns={c: f'A_{c}' for c in feature_cols})

        df_merged = pd.merge(df, h_stats, left_on=['Date','HomeTeam'], right_on=['Date','Team'], how='left').drop(columns=['Team'])
        df_merged = pd.merge(df_merged, a_stats, left_on=['Date','AwayTeam'], right_on=['Date','Team'], how='left').drop(columns=['Team'])

        return df_merged.dropna()

    def _calc_elo_and_h2h(self, df):
        """Calculates Elo ratings and decay-weighted H2H history."""
        elo = EloTracker(k_factor=self.config['elo_k'], home_adv=self.config['elo_home_adv'])
        features_rows = []

        df = df.sort_values('Date').reset_index(drop=True)

        for i, row in df.iterrows():
            date, h, a, res = row['Date'], row['HomeTeam'], row['AwayTeam'], row['FTR']

            # 1. Get PRE-MATCH Ratings
            h_elo = elo.get_rating(h)
            a_elo = elo.get_rating(a)

            # 2. Update Elo for Next Time
            gd = abs(row['FTHG'] - row['FTAG'])
            elo.update(h, a, res, gd)

            # 3. Weighted H2H
            mask = ((df['HomeTeam']==h) & (df['AwayTeam']==a)) | ((df['HomeTeam']==a) & (df['AwayTeam']==h))
            h2h_matches = df[mask & (df['Date'] < date)].tail(5)

            h2h_score = 0
            decay_weight = 1.0

            for _, r in h2h_matches.iloc[::-1].iterrows():
                pts = 0
                if r['FTR'] == 'D': pts = 1
                elif (r['HomeTeam']==h and r['FTR']=='H') or (r['AwayTeam']==h and r['FTR']=='A'): pts = 3
                h2h_score += (pts * decay_weight)
                decay_weight *= self.config['h2h_decay']

            features_rows.append({'H_Elo': h_elo, 'A_Elo': a_elo, 'H2H_Weighted': h2h_score})

        feat_df = pd.DataFrame(features_rows, index=df.index)
        return pd.concat([df, feat_df], axis=1)

    def feature_engineering(self):
        print(f"⚙️ [{self.league_code}] Engineering Features (Twin-Engine Setup)...")
        df = self.data.copy()

        # Execute Modular Steps
        df = self._calc_implied_odds(df)
        df = self._calc_poisson_math(df)
        df = self._calc_rolling_stats(df)
        df = self._calc_elo_and_h2h(df)

        # --- FINAL DERIVED FEATURES ---
        df['Diff_Elo'] = df['H_Elo'] - df['A_Elo']
        df['Abs_Diff_Elo'] = abs(df['H_Elo'] - df['A_Elo'])

        df['Diff_ShotDom'] = (df['H_Roll_Adj_SF']/(df['H_Roll_Adj_SF']+df['H_Roll_SA']+0.1)) - \
                             (df['A_Roll_Adj_SF']/(df['A_Roll_Adj_SF']+df['A_Roll_SA']+0.1))

        df['Diff_SOTDom'] = (df['H_Roll_STF']/(df['H_Roll_STF']+df['H_Roll_STA']+0.1)) - \
                            (df['A_Roll_STF']/(df['A_Roll_STF']+df['A_Roll_STA']+0.1))

        df['Diff_CornDom'] = (df['H_Roll_CF']/(df['H_Roll_CF']+df['H_Roll_CA']+0.1)) - \
                             (df['A_Roll_CF']/(df['A_Roll_CF']+df['A_Roll_CA']+0.1))

        df['Diff_SpecificForm'] = df['H_Specific_Form'] - df['A_Specific_Form']
        df['Diff_Volatility'] = df['H_Roll_GF_Std'] - df['A_Roll_GF_Std']
        df['Diff_Aggression'] = df['H_Roll_Fouls'] - df['A_Roll_Fouls']
        df['Diff_Discipline'] = df['H_Roll_DiscPoints'] - df['A_Roll_DiscPoints']
        df['Boredom_Score'] = (df['H_Roll_TotalGoals'] + df['A_Roll_TotalGoals']) / 2

        df['Market_Elo_Div'] = df['Imp_H'] - (1 / (1 + 10 ** ((-df['Diff_Elo']-75)/400)))

        # --- FEATURE SET DEFINITIONS ---
        self.features_rebel = [
            'Diff_Elo', 'Diff_ShotDom', 'Diff_SOTDom', 'Diff_CornDom',
            'Diff_SpecificForm', 'Diff_Volatility', 'Diff_Aggression', 'Diff_Discipline',
            'Abs_Diff_Elo', 'Boredom_Score', 'H2H_Weighted', 'H_Elo', 'A_Elo'
        ]

        self.features_anchor = self.features_rebel + [
            'Market_Elo_Div', 'Imp_H', 'Imp_D', 'Imp_A',
            'Math_Prob_H', 'Math_Prob_D', 'Math_Prob_A'
        ]

        # SAVE FULL DATA (For lookup during prediction)
        self.feat_data = df.copy()

        # SLICE FOR TRAINING
        self.proc_data = df[self.features_anchor + ['Target', 'Season_ID', 'Date']]
        print(f"✅ [{self.league_code}] Hybrid Sets Ready: Rebel ({len(self.features_rebel)} feats) | Anchor ({len(self.features_anchor)} feats)")

    # -------------------------------------------------------------------------
    # C. TRAINING & TUNING
    # -------------------------------------------------------------------------
    def calculate_decay_weights(self, season_ids):
        max_season = season_ids.max()
        return self.config['decay_alpha'] ** (max_season - season_ids.values)

    def tune_hyperparameters(self, features, y, season_ids, n_trials=10):
        print(f"   🔎 Tuning hyperparameters ({n_trials} trials)...")

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 600),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'objective': 'multi:softprob',
                'num_class': 3,
                'n_jobs': -1,
                'random_state': 42,
                'verbosity': 0
            }

            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_index, test_index in tscv.split(features):
                X_tr, X_val = features.iloc[train_index], features.iloc[test_index]
                y_tr, y_val = y.iloc[train_index], y.iloc[test_index]
                w_tr = self.calculate_decay_weights(season_ids.iloc[train_index])

                scaler = RobustScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)

                model = XGBClassifier(**param)
                model.fit(X_tr_s, y_tr, sample_weight=w_tr)
                preds = model.predict(X_val_s)
                scores.append(f1_score(y_val, preds, average='macro'))

            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        print(f"   ✨ Best Params: {study.best_params}")
        return study.best_params

    def train_models(self, perform_tuning=False):
        print(f"\n🧠 [{self.league_code}] Training Twin Engines (Target: Macro F1)...")

        y = self.proc_data['Target']
        season_ids = self.proc_data['Season_ID']
        split = int(len(self.proc_data) * self.config['split_ratio'])

        w_train = self.calculate_decay_weights(season_ids.iloc[:split])
        y_train = y.iloc[:split]; y_test = y.iloc[split:]

        def train_engine(name, feature_cols):
            print(f"   -> Training {name} Model...")
            X = self.proc_data[feature_cols]
            X_train = X.iloc[:split]; X_test = X.iloc[split:]

            if perform_tuning:
                best_params = self.tune_hyperparameters(X_train, y_train, season_ids.iloc[:split])
            else:
                best_params = self.config['xgb_params']

            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = XGBClassifier(**best_params)
            model.fit(X_train_s, y_train, sample_weight=w_train)

            y_proba = model.predict_proba(X_test_s)
            best_f1, best_thresh = 0, 0.30

            for thresh in np.arange(0.25, 0.35, 0.01):
                preds = []
                for row in y_proba:
                    if row[1] > thresh: preds.append(1)
                    else: preds.append(2 if row[2] > row[0] else 0)
                score = f1_score(y_test, preds, average='macro')
                if score > best_f1:
                    best_f1 = score
                    best_thresh = thresh

            print(f"      {name} Threshold: {best_thresh:.3f} | F1: {best_f1:.4f}")
            return model, scaler, best_thresh

        self.model_anchor, self.scaler_anchor, self.thresh_anchor = train_engine("ANCHOR", self.features_anchor)
        self.model_rebel, self.scaler_rebel, self.thresh_rebel = train_engine("REBEL", self.features_rebel)
    
    def get_history(self, n=10):
        # Extracts last n games and calculates Anchor/Rebel for them
        if self.proc_data is None: return []
        
        last_games = self.feat_data.tail(n).copy()
        history = []
        
        for idx, row in last_games.iterrows():
            # 1. Prepare REBEL Prediction
            f_rebel = pd.DataFrame([row], columns=self.features_rebel)
            f_rebel = f_rebel.reindex(columns=self.features_rebel, fill_value=0)
            probs_reb = self.model_rebel.predict_proba(self.scaler_rebel.transform(f_rebel))[0]
            
            # 2. Prepare ANCHOR Prediction
            f_anchor = pd.DataFrame([row], columns=self.features_anchor)
            f_anchor = f_anchor.reindex(columns=self.features_anchor, fill_value=0)
            probs_anc = self.model_anchor.predict_proba(self.scaler_anchor.transform(f_anchor))[0]

            # 3. Determine Winners
            win_reb = np.argmax(probs_reb) # 0=A, 1=D, 2=H
            win_anc = np.argmax(probs_anc)

            def get_lbl(code): return "HOME" if code == 2 else ("AWAY" if code == 0 else "DRAW")
            
            history.append({
                'home_team': row['HomeTeam'], # Needed for Trinity lookup
                'away_team': row['AwayTeam'],
                'date': row['Date'].strftime('%d-%m'),
                'fixture': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                'result': f"{row['FTHG']:.0f}-{row['FTAG']:.0f} ({row['FTR']})",
                'pred_rebel': get_lbl(win_reb),
                'pred_anchor': get_lbl(win_anc),
                'actual_code': self.target_map[row['FTR']]
            })
            
        return history[::-1]

# =============================================================================
# FLASK INTERFACE BLOCK (Multi-League Updated)
# =============================================================================

# Key = League Code, Value = HybridPipeline instance
_pipelines = {}

def reset_pipeline():
    global _pipelines
    print("TWIN ENGINE SYSTEM WIDE RESET: Clearing all league memory...")
    _pipelines = {}

def get_history_data(league_code='E0'):
    global _pipelines
    if league_code not in _pipelines: return []
    return _pipelines[league_code].get_history(10)

def get_model_2_prediction(home_team, away_team, home_odds, draw_odds, away_odds, league_code='E0'):
    global _pipelines
    
    # Initialize pipeline for this specific league if not already present
    if league_code not in _pipelines:
        print(f"Initializing Twin-Engine Model for League: {league_code}...")
        pipeline = HybridPipeline(league_code=league_code)
        
        success = pipeline.fetch_data()
        if not success:
            print(f"Failed to load data for {league_code} in Model 2")
            return None
            
        pipeline.feature_engineering()
        pipeline.train_models(perform_tuning=False)
        _pipelines[league_code] = pipeline
    
    ai = _pipelines[league_code]
    
    try:
        # 1. Get Team Stats (Logic from class)
        def get_team_stats(team):
            team_games = ai.feat_data[(ai.feat_data['HomeTeam'] == team) | (ai.feat_data['AwayTeam'] == team)]
            if team_games.empty: return None
            last_idx = team_games.last_valid_index()
            last_row = ai.feat_data.loc[last_idx]
            was_home = (last_row['HomeTeam'] == team)
            prefix = 'H_' if was_home else 'A_'
            return {
                'Adj_SF': last_row.get(f'{prefix}Roll_Adj_SF', 5.0),
                'SA': last_row.get(f'{prefix}Roll_SA', 5.0),
                'STF': last_row.get(f'{prefix}Roll_STF', 3.0),
                'STA': last_row.get(f'{prefix}Roll_STA', 3.0),
                'CF': last_row.get(f'{prefix}Roll_CF', 4.0),
                'CA': last_row.get(f'{prefix}Roll_CA', 4.0),
                'SpecForm': last_row.get(f'{prefix}Specific_Form', 1.0),
                'GF_Std': last_row.get(f'{prefix}Roll_GF_Std', 0.5),
                'Fouls': last_row.get(f'{prefix}Roll_Fouls', 10.0),
                'Disc': last_row.get(f'{prefix}Roll_DiscPoints', 1.0),
                'TotG': last_row.get(f'{prefix}Roll_TotalGoals', 2.5)
            }

        h_s = get_team_stats(home_team)
        a_s = get_team_stats(away_team)
        if not h_s or not a_s: return None

        # 2. Update Elo
        elo = EloTracker(k_factor=ai.config['elo_k'], home_adv=ai.config['elo_home_adv'])
        for _, row in ai.data.sort_values('Date').iterrows():
            elo.update(row['HomeTeam'], row['AwayTeam'], row['FTR'], abs(row['FTHG']-row['FTAG']))
        h_elo = elo.get_rating(home_team)
        a_elo = elo.get_rating(away_team)

        # 3. Poisson Calculation
        prob_h, prob_a = 1/home_odds, 1/away_odds
        mu_h = max(0.1, 1.35 + (prob_h - prob_a) * 1.5)
        mu_a = max(0.1, 1.15 + (prob_a - prob_h) * 1.0)
        p_home, p_draw, p_away = 0,0,0
        for h in range(6):
            for a in range(6):
                p = poisson.pmf(h, mu_h) * poisson.pmf(a, mu_a)
                if h>a: p_home+=p
                elif h==a: p_draw+=p
                else: p_away+=p

        # 4. H2H Logic
        mask = ((ai.data['HomeTeam']==home_team) & (ai.data['AwayTeam']==away_team)) | \
               ((ai.data['HomeTeam']==away_team) & (ai.data['AwayTeam']==home_team))
        h2h_matches = ai.data[mask].tail(5)
        h2h_score = 0
        decay = 1.0
        for _, r in h2h_matches.iloc[::-1].iterrows():
            pts = 0
            if r['FTR']=='D': pts=1
            elif (r['HomeTeam']==home_team and r['FTR']=='H') or (r['AwayTeam']==home_team and r['FTR']=='A'): pts=3
            h2h_score += (pts * decay)
            decay *= ai.config['h2h_decay']

        # 5. Build Feature Vector
        f = {}
        f['Diff_Elo'] = h_elo - a_elo
        f['Abs_Diff_Elo'] = abs(h_elo - a_elo)
        f['Diff_ShotDom'] = (h_s['Adj_SF']/(h_s['Adj_SF']+h_s['SA']+0.1)) - (a_s['Adj_SF']/(a_s['Adj_SF']+a_s['SA']+0.1))
        f['Diff_SOTDom'] = (h_s['STF']/(h_s['STF']+h_s['STA']+0.1)) - (a_s['STF']/(a_s['STF']+a_s['STA']+0.1))
        f['Diff_CornDom'] = (h_s['CF']/(h_s['CF']+h_s['CA']+0.1)) - (a_s['CF']/(a_s['CF']+a_s['CA']+0.1))
        f['Diff_SpecificForm'] = h_s['SpecForm'] - a_s['SpecForm']
        f['Diff_Volatility'] = h_s['GF_Std'] - a_s['GF_Std']
        f['Diff_Aggression'] = h_s['Fouls'] - a_s['Fouls']
        f['Diff_Discipline'] = h_s['Disc'] - a_s['Disc']
        f['Boredom_Score'] = (h_s['TotG'] + a_s['TotG']) / 2
        f['H2H_Weighted'] = h2h_score
        f['H_Elo'] = h_elo
        f['A_Elo'] = a_elo
        f['Market_Elo_Div'] = (1/home_odds) - (1 / (1 + 10 ** ((-f['Diff_Elo']-75)/400)))
        f['Imp_H'] = 1/home_odds
        f['Imp_D'] = 1/draw_odds
        f['Imp_A'] = 1/away_odds
        f['Math_Prob_H'] = p_home
        f['Math_Prob_D'] = p_draw
        f['Math_Prob_A'] = p_away

        # 6. Predict Both Engines
        # REBEL
        row_rebel = pd.DataFrame([f], columns=ai.features_rebel)
        probs_rebel = ai.model_rebel.predict_proba(ai.scaler_rebel.transform(row_rebel))[0]

        # ANCHOR
        row_anchor = pd.DataFrame([f], columns=ai.features_anchor)
        probs_anchor = ai.model_anchor.predict_proba(ai.scaler_anchor.transform(row_anchor))[0]

        return {
            "anchor": { "home": probs_anchor[2], "draw": probs_anchor[1], "away": probs_anchor[0] },
            "rebel": { "home": probs_rebel[2], "draw": probs_rebel[1], "away": probs_rebel[0] }
        }

    except Exception as e:
        print(f"Model 2 Error ({league_code}): {e}")
        return None
