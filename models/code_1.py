import pandas as pd
import numpy as np
import requests
import io
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============================================
# GOD MODE BACKEND (DYNAMIC LEAGUE VERSION)
# ============================================

class GodModeEngine:
    def __init__(self, league_code='E0'):
        self.league_code = league_code
        # Historical seasons to train on
        self.SEASONS = ['1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526']
        self.ODDS_URL = f"https://www.football-data.co.uk/mmz4281/{{}}/{self.league_code}.csv"
        
        self.master_df = None
        self.model = None
        self.scaler = None
        self.features = ['Elo_Diff', 'EMA_SOT_Diff', 'EMA_Corn_Diff', 'Eff_Trend_Diff']
        self.current_teams = []
        self.curr_elo_dict = {}

    def load_data(self):
        dfs = []
        for s in self.SEASONS:
            try:
                # Format URL with season code
                url = self.ODDS_URL.format(s)
                
                # Fetch data
                r = requests.get(url)
                if r.status_code != 200: continue

                c = r.content
                df = pd.read_csv(io.StringIO(c.decode('latin-1')))
                df = df.dropna(how='all')
                
                # Capture current season teams for validation
                if s in ['2425', '2526']: 
                    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()
                    self.current_teams = sorted(teams)
                
                # Standardize Columns
                cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','HS','AS','HST','AST','HC','AC']
                df = df[[c for c in cols if c in df.columns]]
                dfs.append(df)
            except: pass
        
        if not dfs: return False
        
        df = pd.concat(dfs, ignore_index=True)
        col_map = {'Date':'date', 'HomeTeam':'home_team', 'AwayTeam':'away_team', 
                   'FTHG':'home_goals', 'FTAG':'away_goals', 
                   'HST':'home_shots_on_target', 'AST':'away_shots_on_target', 
                   'HC':'home_corners', 'AC':'away_corners'}
        df.rename(columns=col_map, inplace=True)
        
        # --- ROBUST DATE PARSING ---
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Fill missing stats with averages to prevent NaN errors
        for c in ['home_shots_on_target', 'away_shots_on_target', 'home_corners', 'away_corners']:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].mean())
            else:
                df[c] = 0.0 # Fallback if column missing entirely
            
        self.master_df = df
        return True

    def engineer_features(self):
        df = self.master_df.copy()
        
        # 1. ELO CALCULATION
        df['home_elo'] = 1500.0
        df['away_elo'] = 1500.0
        curr_elo = {t: 1500.0 for t in pd.concat([df['home_team'], df['away_team']]).unique()}
        k = 20

        for i, row in df.iterrows():
            h, a = row['home_team'], row['away_team']
            h_elo, a_elo = curr_elo.get(h, 1500), curr_elo.get(a, 1500)
            df.at[i, 'home_elo'] = h_elo
            df.at[i, 'away_elo'] = a_elo

            if row['home_goals'] > row['away_goals']: res = 1
            elif row['home_goals'] == row['away_goals']: res = 0.5
            else: res = 0
            
            dr = h_elo - a_elo
            e_h = 1 / (1 + 10**(-dr/400))
            
            curr_elo[h] += k * (res - e_h)
            curr_elo[a] += k * ((1-res) - (1-e_h))
            
        self.curr_elo_dict = curr_elo 

        # 2. EMA (EXPONENTIAL MOVING AVERAGE) FORM
        def create_stream(df):
            h = df[['date', 'home_team', 'home_goals', 'home_shots_on_target', 'home_corners']].copy()
            h.columns = ['date', 'team', 'goals', 'sot', 'corners']
            a = df[['date', 'away_team', 'away_goals', 'away_shots_on_target', 'away_corners']].copy()
            a.columns = ['date', 'team', 'goals', 'sot', 'corners']
            return pd.concat([h, a]).sort_values(['team', 'date'])

        stream = create_stream(df)
        cols = ['goals', 'sot', 'corners']
        # Calculate EMA shifted by 1 (so we only use PAST data for current row)
        stream_ema = stream.groupby('team')[cols].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
        stream = pd.concat([stream, stream_ema.add_prefix('ema_')], axis=1)

        # Merge EMA back to main dataframe
        df = df.merge(stream[['date', 'team', 'ema_goals', 'ema_sot', 'ema_corners']], 
                      left_on=['date', 'home_team'], right_on=['date', 'team'], how='left').rename(columns={'ema_goals':'h_ema_goals', 'ema_sot':'h_ema_sot', 'ema_corners':'h_ema_corn'}).drop(columns=['team'])
        
        df = df.merge(stream[['date', 'team', 'ema_goals', 'ema_sot', 'ema_corners']], 
                      left_on=['date', 'away_team'], right_on=['date', 'team'], how='left').rename(columns={'ema_goals':'a_ema_goals', 'ema_sot':'a_ema_sot', 'ema_corners':'a_ema_corn'}).drop(columns=['team'])

        # 3. FEATURE DIFFERENTIALS
        df['Elo_Diff'] = df['home_elo'] - df['away_elo']
        df['EMA_SOT_Diff'] = df['h_ema_sot'] - df['a_ema_sot']
        df['EMA_Corn_Diff'] = df['h_ema_corn'] - df['a_ema_corn']
        
        h_eff = df['h_ema_goals'] / (df['h_ema_sot'] + 0.1)
        a_eff = df['a_ema_goals'] / (df['a_ema_sot'] + 0.1)
        df['Eff_Trend_Diff'] = h_eff - a_eff

        # 4. TARGET CREATION (0=Away, 1=Draw, 2=Home)
        conditions = [df['home_goals'] > df['away_goals'], df['home_goals'] == df['away_goals']]
        df['target'] = np.select(conditions, [2, 1], default=0)
        
        self.master_df = df.dropna(subset=self.features).copy()

    def train_trinity_model(self):
        df = self.master_df
        if df.empty: return

        X = df[self.features]
        y = df['target']
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Weighted by recency (more weight to recent games)
        weights = np.exp(np.linspace(0, 4, len(X)))
        
        # TRINITY ENSEMBLE
        lr = LogisticRegression(C=0.05, max_iter=1000)
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        xgb_mod = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, 
                                    objective='multi:softmax', num_class=3, random_state=42, eval_metric='mlogloss')
        
        self.model = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_mod)],
            voting='soft', weights=[1, 1, 3]
        )
        self.model.fit(X_scaled, y, sample_weight=weights)

    def predict_match(self, h_team, a_team):
        # --- CRITICAL FIX: Safe Data Retrieval ---
        # Instead of crashing with iloc[-1] on empty data, check if team exists first
        
        df_h = self.master_df[(self.master_df['home_team'] == h_team) | (self.master_df['away_team'] == h_team)]
        df_a = self.master_df[(self.master_df['home_team'] == a_team) | (self.master_df['away_team'] == a_team)]
        
        if df_h.empty or df_a.empty:
            # print(f"⚠️ Trinity: No history for {h_team} or {a_team} in {self.league_code}")
            return None # Fail gracefully

        row_h = df_h.iloc[-1]
        row_a = df_a.iloc[-1]
        
        def get_stat(row, team, stat): 
            return row[f'h_{stat}'] if row['home_team'] == team else row[f'a_{stat}']
        
        h_elo = self.curr_elo_dict.get(h_team, 1500)
        a_elo = self.curr_elo_dict.get(a_team, 1500)

        # Calculate live features based on most recent stats
        elo_diff = h_elo - a_elo
        sot_diff = get_stat(row_h, h_team, 'ema_sot') - get_stat(row_a, a_team, 'ema_sot')
        corn_diff = get_stat(row_h, h_team, 'ema_corn') - get_stat(row_a, a_team, 'ema_corn')
        
        h_eff = get_stat(row_h, h_team, 'ema_goals')/(get_stat(row_h, h_team, 'ema_sot')+0.1)
        a_eff = get_stat(row_a, a_team, 'ema_goals')/(get_stat(row_a, a_team, 'ema_sot')+0.1)
        eff_diff = h_eff - a_eff
        
        input_vec = pd.DataFrame([[elo_diff, sot_diff, corn_diff, eff_diff]], columns=self.features)
        
        if self.scaler is None: return None
        input_scaled = self.scaler.transform(input_vec)
        
        probs = self.model.predict_proba(input_scaled)[0]
        return {'A': probs[0], 'D': probs[1], 'H': probs[2], 'H_Elo': int(h_elo), 'A_Elo': int(a_elo)}

# ============================================
# INTERFACE BLOCK (Multi-League Support)
# ============================================

# Store engines for different leagues in memory
_engines = {}

def get_model_1_prediction(home_team, away_team, league_code='E0'):
    """
    Main entry point. Handles engine initialization and prediction.
    """
    global _engines
    
    # 1. Initialize Engine if missing
    if league_code not in _engines:
        print(f"Initializing Trinity Engine for League: {league_code}...")
        engine = GodModeEngine(league_code=league_code)
        
        success = engine.load_data()
        if not success:
            print(f"❌ Failed to load data for {league_code}")
            return None
            
        engine.engineer_features()
        engine.train_trinity_model()
        _engines[league_code] = engine
        print(f"✅ Engine {league_code} Ready.")

    # 2. Predict
    try:
        engine = _engines[league_code]
        pred = engine.predict_match(home_team, away_team)
        
        if pred is None: return None

        return {
            "home_prob": pred['H'],
            "draw_prob": pred['D'],
            "away_prob": pred['A']
        }
    except Exception as e:
        print(f"Trinity Runtime Error ({league_code}): {e}")
        return None
