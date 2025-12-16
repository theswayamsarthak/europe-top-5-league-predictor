import requests
import sys
import os
import json
import time
import re
import numpy as np
from datetime import datetime

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
try:
    from models import code_1
    from models import code_2
except ImportError:
    import code_1
    import code_2

# --- CONFIG ---
API_KEY = 'ef34aceb66c1cd96711aa1ca8fdde0bc' 

LEAGUE_CONFIG = {
    'E0':  {'sport': 'soccer_epl',           'region': 'uk', 'name': 'Premier League'},
    'SP1': {'sport': 'soccer_spain_la_liga', 'region': 'eu', 'name': 'La Liga'},
    'D1':  {'sport': 'soccer_germany_bundesliga', 'region': 'eu', 'name': 'Bundesliga'},
    'I1':  {'sport': 'soccer_italy_serie_a', 'region': 'eu', 'name': 'Serie A'},
    'F1':  {'sport': 'soccer_france_ligue_one', 'region': 'eu', 'name': 'Ligue 1'}
}

MARKETS = 'h2h'
STATS_FILE = 'model_stats.json'
ARCHIVE_FILE = 'predictions_archive.json' # <--- NEW: The Permanent Ledger
CACHE_DURATION = 10 * 24 * 60 * 60  # 10 Days

# --- MASTER TEAM MAPPING ---
TEAM_MAP = {
    # --- GERMANY (Bundesliga) ---
    'vfl wolfsburg': 'Wolfsburg',
    'hamburger sv': 'Hamburg',
    'fsv mainz 05': 'Mainz',
    'mainz 05': 'Mainz',
    'rb leipzig': 'Leipzig',
    'koln': 'FC Koln',
    '1. fc koln': 'FC Koln',
    'fc koln': 'FC Koln',
    'borussia monchengladbach': 'M\'gladbach',
    'monchengladbach': 'M\'gladbach',
    'm\'gladbach': 'M\'gladbach',
    'bayer leverkusen': 'Leverkusen',
    'borussia dortmund': 'Dortmund',
    'heidenheim': 'Heidenheim',
    'rb leipzig': 'RB Leipzig',
    'r.b. leipzig': 'RB Leipzig',
    'rasenballsport leipzig': 'RB Leipzig',
    'rasen ballsport leipzig': 'RB Leipzig',
    'red bull leipzig': 'RB Leipzig',
    'leipzig': 'RB Leipzig',
    'st pauli': 'St Pauli',
    'st. pauli': 'St Pauli',
    'holstein kiel': 'Holstein Kiel',
    'eintracht frankfurt': 'Ein Frankfurt',
    'tsg hoffenheim': 'Hoffenheim',
    'hoffenheim': 'Hoffenheim',
    'vfb stuttgart': 'Stuttgart',
    'werder bremen': 'Werder Bremen',
    'augsburg': 'Augsburg',
    'union berlin': 'Union Berlin',
    'bochum': 'Bochum',
    'wolfsburg': 'Wolfsburg',
    'bayern munich': 'Bayern Munich',
    'darmstadt': 'Darmstadt',
    'freiburg': 'Freiburg',
    'hamburg': 'Hamburg',

    # --- SPAIN (La Liga) ---
    'deportivo alaves': 'Alaves',
    'alavés': 'Alaves',
    'alaves': 'Alaves',
    'rcd espanyol': 'Espanol',
    'espanyol': 'Espanol',
    'rayo vallecano': 'Vallecano',
    'vallecano': 'Vallecano',
    'athletic bilbao': 'Ath Bilbao',
    'atletico madrid': 'Ath Madrid', 
    'ca osasuna': 'Osasuna', 
    'elche cf': 'Elche', 
    'real betis': 'Betis', 
    'real sociedad': 'Sociedad',
    'celta vigo': 'Celta', 
    'rcd mallorca': 'Mallorca',
    'girona': 'Girona',
    'valencia': 'Valencia',
    'villarreal': 'Villarreal',
    'sevilla': 'Sevilla',
    'cadiz': 'Cadiz',
    'granada': 'Granada',
    'las palmas': 'Las Palmas',
    'almeria': 'Almeria',
    'real madrid': 'Real Madrid',
    'barcelona': 'Barcelona',
    'getafe': 'Getafe',
    'levante': 'Levante',
    'oviedo': 'Oviedo',

    # --- FRANCE (Ligue 1) ---
    'paris saint germain': 'Paris SG', 
    'paris sg': 'Paris SG', 
    'psg': 'Paris SG',
    'saint-etienne': 'St Etienne', 
    'st etienne': 'St Etienne',
    'as monaco': 'Monaco',
    'olympique marseille': 'Marseille',
    'olympique lyonnais': 'Lyon',
    'losc lille': 'Lille',
    'ogc nice': 'Nice',
    'stade rennes': 'Rennes',
    'rennes': 'Rennes',
    'rc lens': 'Lens',
    'stade reims': 'Reims',
    'montpellier': 'Montpellier',
    'strasbourg': 'Strasbourg',
    'nantes': 'Nantes',
    'toulouse': 'Toulouse',
    'le havre': 'Le Havre',
    'brest': 'Brest',
    'metz': 'Metz',
    'clermont': 'Clermont',
    'lorient': 'Lorient',
    'auxerre': 'Auxerre',
    'angers': 'Angers',

    # --- ITALY (Serie A) ---
    'inter milan': 'Inter',
    'ac milan': 'Milan',
    'as roma': 'Roma',
    'hellas verona': 'Verona',
    'juventus': 'Juventus',
    'lazio': 'Lazio',
    'napoli': 'Napoli',
    'atalanta': 'Atalanta',
    'fiorentina': 'Fiorentina',
    'torino': 'Torino',
    'udinese': 'Udinese',
    'bologna': 'Bologna',
    'monza': 'Monza',
    'lecce': 'Lecce',
    'empoli': 'Empoli',
    'salernitana': 'Salernitana',
    'sassuolo': 'Sassuolo',
    'frosinone': 'Frosinone',
    'genoa': 'Genoa',
    'cagliari': 'Cagliari',
    'parma': 'Parma',
    'como': 'Como',
    'venezia': 'Venezia',
    'atalanta bc': 'Atalanta',
    'atalanta': 'Atalanta',
    
    # --- ENGLAND ---
    'manchester united': 'Man United',
    'manchester city': 'Man City',
    'tottenham hotspur': 'Tottenham',
    'newcastle united': 'Newcastle',
    'nottingham forest': 'Nott\'m Forest',
    'wolverhampton wanderers': 'Wolves',
    'leicester city': 'Leicester',
    'leeds united': 'Leeds',
    'west ham united': 'West Ham',
    'brighton and hove albion': 'Brighton',
    'sheffield united': 'Sheffield United',
    'luton town': 'Luton',
    'ipswich town': 'Ipswich'
}

class ModelManager:
    def __init__(self):
        # Load the Archive into memory when Manager starts
        self.archive = self._load_archive()

    def _normalize_name(self, name):
        if not isinstance(name, str): return ""
        clean_name = name.lower().strip()
        clean_name = re.sub(r'^(1\.?\s?fc|fc|sc|sv|rc|rcd)\s+', '', clean_name)
        clean_name = re.sub(r'\s+(fc|cf|sc)$', '', clean_name)
        replacements = {
            'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a',
            'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e',
            'í': 'i', 'ì': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o',
            'ú': 'u', 'ù': 'u', 'ü': 'u',
            'ñ': 'n', 'ç': 'c', 'ß': 'ss'
        }
        for char, rep in replacements.items():
            clean_name = clean_name.replace(char, rep)
        clean_name = clean_name.strip()
        final_name = TEAM_MAP.get(clean_name, TEAM_MAP.get(name.lower(), name))
        return final_name

    # --- NEW ARCHIVE METHODS ---
    def _load_archive(self):
        if os.path.exists(ARCHIVE_FILE):
            try:
                with open(ARCHIVE_FILE, 'r') as f:
                    return json.load(f)
            except: return {}
        return {}

    def _save_to_archive(self, game_id, predictions):
        """
        Saves predictions. INTELLIGENT MERGE:
        If entry exists, it MERGES new keys (like anchor/rebel) 
        but does NOT overwrite existing ones (like trinity).
        """
        updated = False
        
        # 1. Create entry if it doesn't exist
        if game_id not in self.archive:
            self.archive[game_id] = {}
        
        existing = self.archive[game_id]
        
        # 2. Merge new predictions into existing entry
        for key, value in predictions.items():
            # Only write if value is valid and (key missing OR current is N/A)
            if value != "N/A" and (key not in existing or existing[key] == "N/A"):
                existing[key] = value
                updated = True
        
        self.archive[game_id] = existing

        # 3. Write to disk only if data changed
        if updated:
            with open(ARCHIVE_FILE, 'w') as f:
                json.dump(self.archive, f, indent=4)

    def get_dashboard_data(self, league_code='E0'):
        cache_file = f"dashboard_cache_{league_code}.json"

        # 1. CHECK CACHE
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    if (time.time() - cached['timestamp']) < CACHE_DURATION:
                        print(f"Manager: Loading {league_code} from CACHE.")
                        return cached['data']
            except Exception as e:
                print(f"Cache Read Error ({league_code}): {e}")

        # 2. RUN ANALYSIS
        print(f"Manager: Cache expired for {league_code}. Running Analysis...")
        data = self._generate_fresh_data(league_code)
        
        # 3. SAVE CACHE
        if data:
            try:
                with open(cache_file, 'w') as f:
                    json.dump({'timestamp': time.time(), 'data': data}, f)
            except Exception as e:
                print(f"Cache Write Error ({league_code}): {e}")
        
        return data

    def _generate_fresh_data(self, league_code):
        if league_code not in LEAGUE_CONFIG: return {'live': [], 'history': []}

        config = LEAGUE_CONFIG[league_code]
        # Fetching raw odds
        raw_odds = self._fetch_odds_api(config['sport'], config['region'])
        live_games = []

        if raw_odds:
            print(f"Processing {len(raw_odds)} live games for {league_code}...")
            
            # Optional limit
            # raw_odds = raw_odds[:10]

            for game in raw_odds:
                home = self._normalize_name(game['home_team'])
                away = self._normalize_name(game['away_team'])
                
                h_odd, d_odd, a_odd = 0, 0, 0
                if game['bookmakers']:
                    bk = next((b for b in game['bookmakers'] if b['key'] == 'bet365'), game['bookmakers'][0])
                    for out in bk['markets'][0]['outcomes']:
                        if out['name'] == game['home_team']: h_odd = out['price']
                        elif out['name'] == game['away_team']: a_odd = out['price']
                        elif out['name'] == 'Draw': d_odd = out['price']
                
                if h_odd == 0: continue

                item = {
                    'date': game['commence_time'].split('T')[0],
                    'home_team': home, 'away_team': away,
                    'odds_h': h_odd, 'odds_d': d_odd, 'odds_a': a_odd,
                    'trinity_h': '--', 'trinity_a': '--',
                    'anchor_h': '--', 'anchor_a': '--',
                    'rebel_h': '--', 'rebel_a': '--',
                    'verdict': "No Value", 'verdict_class': "neutral"
                }

                # --- GENERATE & ARCHIVE PREDICTIONS ---
                try:
                    p1 = code_1.get_model_1_prediction(home, away, league_code=league_code)
                    p2 = code_2.get_model_2_prediction(home, away, h_odd, d_odd, a_odd, league_code=league_code)

                    # Prepare Archive Data
                    archive_data = {}

                    # 1. TRINITY PREDICTION
                    if p1:
                        item['trinity_h'] = f"{p1['home_prob']:.1%}"
                        item['trinity_a'] = f"{p1['away_prob']:.1%}"
                        
                        probs = [p1['away_prob'], p1['draw_prob'], p1['home_prob']]
                        winner_idx = np.argmax(probs)
                        archive_data['pred_trinity'] = "HOME" if winner_idx == 2 else ("AWAY" if winner_idx == 0 else "DRAW")

                    # 2. ANCHOR & REBEL PREDICTIONS
                    if p2:
                        anc, reb = p2['anchor'], p2['rebel']
                        item['anchor_h'] = f"{anc['home']:.1%}"
                        item['anchor_a'] = f"{anc['away']:.1%}"
                        item['rebel_h'] = f"{reb['home']:.1%}"
                        item['rebel_a'] = f"{reb['away']:.1%}"

                        # Helper to convert probabilities to "HOME", "AWAY", "DRAW"
                        def get_winner(probs_dict):
                            # probs_dict is {'home': 0.5, 'away': 0.3, 'draw': 0.2}
                            h, a, d = probs_dict['home'], probs_dict['away'], probs_dict['draw']
                            if h > a and h > d: return "HOME"
                            if a > h and a > d: return "AWAY"
                            return "DRAW"

                        archive_data['pred_anchor'] = get_winner(anc)
                        archive_data['pred_rebel'] = get_winner(reb)

                        # Logic for Verdict (Value Betting)
                        imp_h, imp_a = 1/h_odd, 1/a_odd
                        edge_h = reb['home'] - imp_h
                        edge_a = reb['away'] - imp_a
                        
                        if edge_h > 0.05:
                            item['verdict'] = f"HOME VALUE (+{edge_h*100:.1f}%)"
                            item['verdict_class'] = "value-home"
                        elif edge_a > 0.05:
                            item['verdict'] = f"AWAY VALUE (+{edge_a*100:.1f}%)"
                            item['verdict_class'] = "value-away"
                        elif reb['draw'] > 0.29:
                            item['verdict'] = "DRAW WATCH"
                            item['verdict_class'] = "value-draw"

                    # --- SAVE ALL THREE TO ARCHIVE (MERGE MODE) ---
                    unique_id = f"{league_code}-{home}-{away}"
                    self._save_to_archive(unique_id, archive_data)

                except Exception as e:
                    print(f"⚠️ Prediction Error for {home} vs {away}: {e}")
                    item['verdict'] = "Data Error"

                live_games.append(item)

        # 3. History & Stats
        history = self._get_full_history(league_code)
        self._update_accuracy_stats(history, league_code)

        return {
            'live': live_games, 
            'history': history, 
            'league_name': config['name'] 
        }

    def _fetch_odds_api(self, sport, region):
        url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
        params = {'api_key': API_KEY, 'regions': region, 'markets': MARKETS, 'oddsFormat': 'decimal'}
        try:
            resp = requests.get(url, params=params)
            return resp.json() if resp.status_code == 200 else []
        except: return []

    def _get_full_history(self, league_code):
        try:
            raw_history = code_2.get_history_data(league_code=league_code) 
            clean_history = []
            
            for h in raw_history:
                home = self._normalize_name(h['home_team'])
                away = self._normalize_name(h['away_team'])
                
                # Unique ID for Lookup
                unique_id = f"{league_code}-{home}-{away}"
                
                # Defaults
                trinity_res = "N/A"
                anchor_res = "N/A"
                rebel_res = "N/A"
                
                # --- STRATEGY: STRICT ARCHIVE LOOKUP ONLY ---
                
                # 1. Check Archive (The Honest Way)
                if unique_id in self.archive:
                    saved = self.archive[unique_id]
                    trinity_res = saved.get('pred_trinity', "N/A")
                    anchor_res = saved.get('pred_anchor', "N/A")
                    rebel_res = saved.get('pred_rebel', "N/A")

                # 2. NO FALLBACKS
                # If it's not in the archive, it stays "N/A".
                # This ensures your history log is PURE. No fake calculations.

                clean_history.append({
                    'date': h['date'], 
                    'fixture': h['fixture'], 
                    'result': h['result'],
                    'pred_trinity': trinity_res, 
                    'pred_anchor': anchor_res, 
                    'pred_rebel': rebel_res
                })
            return clean_history
        except Exception as e: 
            print(f"History Error ({league_code}): {e}")
            return []

    def _update_accuracy_stats(self, history_games, league_code):
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r') as f: stats = json.load(f)
            except: stats = self._get_empty_stats()
        else: stats = self._get_empty_stats()

        if league_code not in stats:
            stats[league_code] = {
                "trinity": {"correct":0,"total":0},
                "anchor": {"correct":0,"total":0},
                "rebel": {"correct":0,"total":0}
            }

        updates_made = False
        
        for game in history_games:
            # Create a unique ID specifically for the stats processing
            # Using date here ensures we don't process the same match twice
            game_id = f"{league_code}-{game['date']}-{game['fixture']}"
            
            if game_id in stats['processed_ids']: continue 
            
            # Only process if we have a valid result
            if '?' in game['result']: continue

            updates_made = True
            stats['processed_ids'].append(game_id)
            
            try:
                res_char = game['result'].split('(')[1].replace(')', '') 
                actual = "HOME" if res_char == 'H' else ("AWAY" if res_char == 'A' else "DRAW")
            except: continue

            for model in ['trinity', 'anchor', 'rebel']:
                pred_key = f'pred_{model}'
                prediction = game.get(pred_key, "N/A")
                
                # ONLY COUNT VALID PREDICTIONS
                if prediction != "N/A":
                    stats[league_code][model]['total'] += 1
                    if prediction == actual: 
                        stats[league_code][model]['correct'] += 1
        
        if updates_made:
            stats['last_updated'] = datetime.now().strftime("%d %b %Y, %H:%M")
            # Keep the processed IDs list from growing infinitely
            if len(stats['processed_ids']) > 1000: 
                stats['processed_ids'] = stats['processed_ids'][-1000:]
            with open(STATS_FILE, 'w') as f: json.dump(stats, f)
        
        return stats

    def _get_empty_stats(self):
        return {"last_updated": "Initializing...", "processed_ids": []}

    def clear_cache(self):
        files = [f for f in os.listdir('.') if f.startswith('dashboard_cache_') and f.endswith('.json')]
        for f in files:
            try: os.remove(f)
            except: pass

    def get_stats(self):
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f: return json.load(f)
        return self._get_empty_stats()