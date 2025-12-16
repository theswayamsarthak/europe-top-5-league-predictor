import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
SEASON = "2526" # 2025/2026 Season
URL_BASE = f"https://www.football-data.co.uk/mmz4281/{SEASON}/"

LEAGUES = {
    "E0": "Premier League (UK)",
    "SP1": "La Liga (ESP)",
    "D1": "Bundesliga (GER)",
    "I1": "Serie A (ITA)",
    "F1": "Ligue 1 (FRA)"
}

def fetch_live_results():
    """Pulls the latest results from Football-Data.co.uk for all leagues."""
    print(":: FETCHING LIVE RESULTS FROM SERVER...")
    all_results = []
    
    for code in LEAGUES.keys():
        try:
            url = f"{URL_BASE}{code}.csv"
            df = pd.read_csv(url)
            # Standardize columns
            df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTR']] # FTR = Full Time Result (H,D,A)
            df['League'] = code
            all_results.append(df)
            print(f"   > {code}: OK ({len(df)} matches)")
        except Exception as e:
            print(f"   > {code}: FAILED ({e})")
            
    if not all_results:
        return pd.DataFrame()
        
    # Combine all leagues into one massive reference table
    master_df = pd.concat(all_results, ignore_index=True)
    
    # Convert Date format to match standard (depending on CSV source, usually dd/mm/yyyy)
    # Football-data often uses 2-digit years.
    master_df['Date'] = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce')
    
    return master_df

def calculate_accuracy():
    """Compares history.csv preds with live results."""
    
    # 1. Load Your Predictions
    try:
        my_preds = pd.read_csv('history.csv')
        my_preds['Date'] = pd.to_datetime(my_preds['Date'], dayfirst=True)
    except FileNotFoundError:
        print("ERROR: history.csv not found.")
        return

    # 2. Get Real Results
    real_results = fetch_live_results()
    
    if real_results.empty:
        print("CRITICAL: No result data available.")
        return

    # 3. Merge (Join on HomeTeam + AwayTeam to ensure correct match)
    # Note: Team names MUST match. If 'Man Utd' != 'Man United', this fails. 
    # In a real app, you need a name mapping dictionary.
    merged = pd.merge(my_preds, real_results, on=['HomeTeam', 'AwayTeam'], how='inner', suffixes=('', '_Actual'))

    # 4. Calculate Stats per League
    stats_output = {}
    
    for code, name in LEAGUES.items():
        # Filter for this league
        league_data = merged[merged['League'] == code].copy()
        
        # Sort by date (Oldest -> Newest)
        league_data = league_data.sort_values(by='Date')
        
        if league_data.empty:
            stats_output[code] = {
                "name": name,
                "trinity": {"l10": "N/A", "cum": "0%"},
                "anchor": {"l10": "N/A", "cum": "0%"},
                "rebel": {"l10": "N/A", "cum": "0%"}
            }
            continue

        # Helper to calc %
        def get_acc(series_preds, series_actual):
            # Compare Prediction vs Actual Result (H, D, A)
            correct = (series_preds == series_actual).sum()
            total = len(series_preds)
            if total == 0: return "0%"
            return f"{int((correct/total)*100)}%"

        # --- CUMULATIVE (Expanding Window) ---
        # Takes all matches from start of file to now
        cum_trinity = get_acc(league_data['Trinity_Pred'], league_data['FTR'])
        cum_anchor  = get_acc(league_data['Anchor_Pred'],  league_data['FTR'])
        cum_rebel   = get_acc(league_data['Rebel_Pred'],   league_data['FTR'])

        # --- LAST 10 (Rolling Window) ---
        # Takes only the last 10 rows
        l10_data = league_data.tail(10)
        l10_trinity = get_acc(l10_data['Trinity_Pred'], l10_data['FTR'])
        l10_anchor  = get_acc(l10_data['Anchor_Pred'],  l10_data['FTR'])
        l10_rebel   = get_acc(l10_data['Rebel_Pred'],   l10_data['FTR'])

        stats_output[code] = {
            "name": name,
            "trinity": {"l10": l10_trinity, "cum": cum_trinity},
            "anchor":  {"l10": l10_anchor,  "cum": cum_anchor},
            "rebel":   {"l10": l10_rebel,   "cum": cum_rebel}
        }

    return stats_output

if __name__ == "__main__":
    new_stats = calculate_accuracy()
    print("\n:: COPY THIS DICTIONARY INTO APP.PY ::\n")
    import json
    print(json.dumps(new_stats, indent=4))