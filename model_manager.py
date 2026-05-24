import requests
import sys
import os
import json
import time
import re
import numpy as np
from datetime import datetime
from supabase import create_client, Client

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
try:
    from models import code_1
    from models import code_2
except ImportError:
    import code_1
    import code_2

# --- CONFIG ---
API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    print("WARNING: ODDS_API_KEY not set in environment variables")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: SUPABASE_URL or SUPABASE_KEY not set — persistence disabled")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase connected.")

LEAGUE_CONFIG = {
    'E0':  {'sport': 'soccer_epl',               'region': 'uk', 'name': 'Premier League'},
    'SP1': {'sport': 'soccer_spain_la_liga',      'region': 'eu', 'name': 'La Liga'},
    'D1':  {'sport': 'soccer_germany_bundesliga', 'region': 'eu', 'name': 'Bundesliga'},
    'I1':  {'sport': 'soccer_italy_serie_a',      'region': 'eu', 'name': 'Serie A'},
    'F1':  {'sport': 'soccer_france_ligue_one',   'region': 'eu', 'name': 'Ligue 1'}
}

MARKETS = 'h2h'
CACHE_DURATION = 3 * 60 * 60  # 3 hours

# In-memory cache (lives within one server session)
_memory_cache = {}

# --- MASTER TEAM MAPPING ---
TEAM_MAP = {
    # GERMANY
    'vfl wolfsburg': 'Wolfsburg',
    'hamburger sv': 'Hamburg',
    'fsv mainz 05': 'Mainz',
    'mainz 05': 'Mainz',
    'koln': 'FC Koln',
    '1. fc koln': 'FC Koln',
    'fc koln': 'FC Koln',
    'borussia monchengladbach': "M'gladbach",
    'monchengladbach': "M'gladbach",
    "m'gladbach": "M'gladbach",
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
    # SPAIN
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
    # FRANCE
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
    'paris fc': 'Paris FC',
    # ITALY
    'inter milan': 'Inter',
    'ac milan': 'Milan',
    'as roma': 'Roma',
    'hellas verona': 'Verona',
    'juventus': 'Juventus',
    'lazio': 'Lazio',
    'napoli': 'Napoli',
    'atalanta': 'Atalanta',
    'atalanta bc': 'Atalanta',
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
    'cremonese': 'Cremonese',
    'pisa': 'Pisa',
    # ENGLAND
    'manchester united': 'Man United',
    'manchester city': 'Man City',
    'tottenham hotspur': 'Tottenham',
    'newcastle united': 'Newcastle',
    "nottingham forest": "Nott'm Forest",
    'wolverhampton wanderers': 'Wolves',
    'leicester city': 'Leicester',
    'leeds united': 'Leeds',
    'west ham united': 'West Ham',
    'brighton and hove albion': 'Brighton',
    'sheffield united': 'Sheffield United',
    'luton town': 'Luton',
    'ipswich town': 'Ipswich',
    'sunderland': 'Sunderland',
    'burnley': 'Burnley',
    'aston villa': 'Aston Villa',
    'chelsea': 'Chelsea',
    'arsenal': 'Arsenal',
    'liverpool': 'Liverpool',
    'everton': 'Everton',
    'crystal palace': 'Crystal Palace',
    'brentford': 'Brentford',
    'fulham': 'Fulham',
    'bournemouth': 'Bournemouth',
}


class ModelManager:
    def __init__(self):
        self.archive = self._load_archive()
        self._is_warm = False  # True once all league engines are loaded

    def warm_up(self):
        """
        Pre-warm all five league pipelines in the background so the first
        real user request is not the one that triggers 60-90s of training.
        Called from app.py in a daemon thread at startup.
        """
        print(":: PRE-WARM STARTED — loading all league engines ::")
        for league_code in LEAGUE_CONFIG:
            try:
                print(f"  Warming {league_code}...")
                self._generate_fresh_data(league_code)
                print(f"  {league_code} warm.")
            except Exception as e:
                print(f"  Warm-up error ({league_code}): {e}")
        self._is_warm = True
        print(":: PRE-WARM COMPLETE ::")

    # -------------------------------------------------------------------------
    # NAME NORMALISATION
    # -------------------------------------------------------------------------
    def _normalize_name(self, name):
        if not isinstance(name, str):
            return ""
        clean = name.lower().strip()
        clean = re.sub(r'^(1\.?\s?fc|fc|sc|sv|rc|rcd)\s+', '', clean)
        clean = re.sub(r'\s+(fc|cf|sc)$', '', clean)
        for char, rep in {
            'á':'a','à':'a','ä':'a','â':'a',
            'é':'e','è':'e','ë':'e','ê':'e',
            'í':'i','ì':'i','ï':'i',
            'ó':'o','ò':'o','ö':'o','ô':'o',
            'ú':'u','ù':'u','ü':'u',
            'ñ':'n','ç':'c','ß':'ss'
        }.items():
            clean = clean.replace(char, rep)
        clean = clean.strip()
        return TEAM_MAP.get(clean, TEAM_MAP.get(name.lower(), name))

    # -------------------------------------------------------------------------
    # SUPABASE — PREDICTIONS ARCHIVE
    # -------------------------------------------------------------------------
    # Path for local fallback archive (used when Supabase is not configured)
    _LOCAL_ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), 'predictions_archive.json')

    def _load_archive(self):
        if not supabase:
            # Fallback: load from local JSON file so predictions survive restarts
            try:
                if os.path.exists(self._LOCAL_ARCHIVE_PATH):
                    with open(self._LOCAL_ARCHIVE_PATH, 'r') as f:
                        archive = json.load(f)
                    print(f"Archive loaded from local JSON: {len(archive)} entries.")
                    return archive
            except Exception as e:
                print(f"Local archive load error: {e}")
            return {}
        try:
            result = supabase.table('predictions_archive').select('*').execute()
            archive = {}
            for row in result.data:
                archive[row['game_id']] = {
                    'pred_trinity': row.get('pred_trinity'),
                    'pred_anchor':  row.get('pred_anchor'),
                    'pred_rebel':   row.get('pred_rebel'),
                }
            print(f"Archive loaded: {len(archive)} entries.")
            return archive
        except Exception as e:
            print(f"Archive load error: {e}")
            return {}

    def _save_to_archive(self, game_id, predictions):
        """Merge-write: never overwrites a valid existing prediction."""
        if not supabase:
            # Fallback: persist to local JSON file
            existing = self.archive.get(game_id, {})
            merged = dict(existing)
            updated = False
            for key, value in predictions.items():
                if value and value != 'N/A':
                    if not existing.get(key) or existing[key] == 'N/A':
                        merged[key] = value
                        updated = True
            if updated:
                self.archive[game_id] = merged
                try:
                    with open(self._LOCAL_ARCHIVE_PATH, 'w') as f:
                        json.dump(self.archive, f, indent=2)
                except Exception as e:
                    print(f"Local archive save error: {e}")
            return

        existing = self.archive.get(game_id, {})
        merged = dict(existing)
        updated = False

        for key, value in predictions.items():
            if value and value != 'N/A':
                if not existing.get(key) or existing[key] == 'N/A':
                    merged[key] = value
                    updated = True

        if not updated:
            return

        self.archive[game_id] = merged
        try:
            supabase.table('predictions_archive').upsert({
                'game_id':      game_id,
                'pred_trinity': merged.get('pred_trinity'),
                'pred_anchor':  merged.get('pred_anchor'),
                'pred_rebel':   merged.get('pred_rebel'),
                'updated_at':   datetime.utcnow().isoformat(),
            }).execute()
        except Exception as e:
            print(f"Archive save error ({game_id}): {e}")

    # -------------------------------------------------------------------------
    # SUPABASE — MODEL ACCURACY STATS
    # -------------------------------------------------------------------------
    def _load_stats(self):
        if not supabase:
            return self._empty_stats()
        try:
            result = supabase.table('model_stats').select('*').execute()
            stats = self._empty_stats()
            for row in result.data:
                league = row['league']
                model  = row['model']
                if league in stats:
                    stats[league][model] = {
                        'correct': row.get('correct', 0),
                        'total':   row.get('total', 0),
                    }
            return stats
        except Exception as e:
            print(f"Stats load error: {e}")
            return self._empty_stats()

    def _save_stat(self, league, model, correct_delta, total_delta):
        if not supabase:
            return
        stat_id = f"{league}_{model}"
        try:
            res = supabase.table('model_stats').select('*').eq('stat_id', stat_id).execute()
            if res.data:
                row = res.data[0]
                supabase.table('model_stats').update({
                    'correct':    row['correct'] + correct_delta,
                    'total':      row['total']   + total_delta,
                    'updated_at': datetime.utcnow().isoformat(),
                }).eq('stat_id', stat_id).execute()
            else:
                supabase.table('model_stats').insert({
                    'stat_id':    stat_id,
                    'league':     league,
                    'model':      model,
                    'correct':    correct_delta,
                    'total':      total_delta,
                    'updated_at': datetime.utcnow().isoformat(),
                }).execute()
        except Exception as e:
            print(f"Stat save error ({stat_id}): {e}")

    def _empty_stats(self):
        return {
            code: {
                'trinity': {'correct': 0, 'total': 0},
                'anchor':  {'correct': 0, 'total': 0},
                'rebel':   {'correct': 0, 'total': 0},
            }
            for code in LEAGUE_CONFIG
        }

    # -------------------------------------------------------------------------
    # SUPABASE — PROCESSED RESULTS (prevents double-counting)
    # -------------------------------------------------------------------------
    def _is_processed(self, game_id):
        if not supabase:
            return False
        try:
            res = supabase.table('processed_results').select('game_id').eq('game_id', game_id).execute()
            return len(res.data) > 0
        except Exception as e:
            print(f"Processed check error: {e}")
            return False

    def _mark_processed(self, game_id):
        if not supabase:
            return
        try:
            supabase.table('processed_results').upsert({
                'game_id':      game_id,
                'processed_at': datetime.utcnow().isoformat(),
            }).execute()
        except Exception as e:
            print(f"Mark processed error ({game_id}): {e}")

    # -------------------------------------------------------------------------
    # IN-MEMORY CACHE (3-hour TTL)
    # -------------------------------------------------------------------------
    def _get_cache(self, league_code):
        cached = _memory_cache.get(league_code)
        if cached and (time.time() - cached['timestamp']) < CACHE_DURATION:
            print(f"Serving {league_code} from memory cache.")
            return cached['data']
        return None

    def _set_cache(self, league_code, data):
        _memory_cache[league_code] = {'timestamp': time.time(), 'data': data}

    def clear_cache(self):
        _memory_cache.clear()
        print("Memory cache cleared.")

    # -------------------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------------------
    def get_dashboard_data(self, league_code='E0'):
        cached = self._get_cache(league_code)
        if cached:
            return cached

        print(f"Cache miss for {league_code}. Generating fresh data...")
        data = self._generate_fresh_data(league_code)
        if data:
            self._set_cache(league_code, data)
        return data

    # -------------------------------------------------------------------------
    # CORE DATA GENERATION
    # -------------------------------------------------------------------------
    def _generate_fresh_data(self, league_code):
        if league_code not in LEAGUE_CONFIG:
            return {'live': [], 'history': []}

        config   = LEAGUE_CONFIG[league_code]
        raw_odds = self._fetch_odds_api(config['sport'], config['region'])
        live_games = []

        if raw_odds:
            print(f"Processing {len(raw_odds)} games for {league_code}...")
            for game in raw_odds:
                home = self._normalize_name(game['home_team'])
                away = self._normalize_name(game['away_team'])

                h_odd = d_odd = a_odd = 0
                if game['bookmakers']:
                    bk = next(
                        (b for b in game['bookmakers'] if b['key'] == 'bet365'),
                        game['bookmakers'][0]
                    )
                    for out in bk['markets'][0]['outcomes']:
                        if out['name'] == game['home_team']:   h_odd = out['price']
                        elif out['name'] == game['away_team']: a_odd = out['price']
                        elif out['name'] == 'Draw':            d_odd = out['price']

                if h_odd == 0:
                    continue

                item = {
                    'date': game['commence_time'].split('T')[0],
                    'home_team': home, 'away_team': away,
                    'odds_h': h_odd, 'odds_d': d_odd, 'odds_a': a_odd,
                    'trinity_h': '--', 'trinity_a': '--',
                    'anchor_h':  '--', 'anchor_a':  '--',
                    'rebel_h':   '--', 'rebel_a':   '--',
                    'pred_trinity': 'N/A',
                    'pred_anchor':  'N/A',
                    'pred_rebel':   'N/A',
                    'verdict': 'No Value', 'verdict_class': 'neutral',
                }

                try:
                    p1 = code_1.get_model_1_prediction(home, away, league_code=league_code)
                    p2 = code_2.get_model_2_prediction(home, away, h_odd, d_odd, a_odd, league_code=league_code)
                    archive_data = {}

                    if p1:
                        item['trinity_h'] = f"{p1['home_prob']:.1%}"
                        item['trinity_a'] = f"{p1['away_prob']:.1%}"
                        probs = [p1['away_prob'], p1['draw_prob'], p1['home_prob']]
                        idx = int(np.argmax(probs))
                        call = 'HOME' if idx == 2 else ('AWAY' if idx == 0 else 'DRAW')
                        item['pred_trinity'] = call
                        archive_data['pred_trinity'] = call

                    if p2:
                        anc, reb = p2['anchor'], p2['rebel']
                        item['anchor_h'] = f"{anc['home']:.1%}"
                        item['anchor_a'] = f"{anc['away']:.1%}"
                        item['rebel_h']  = f"{reb['home']:.1%}"
                        item['rebel_a']  = f"{reb['away']:.1%}"

                        def get_winner(d):
                            if d['home'] > d['away'] and d['home'] > d['draw']: return 'HOME'
                            if d['away'] > d['home'] and d['away'] > d['draw']: return 'AWAY'
                            return 'DRAW'

                        item['pred_anchor'] = get_winner(anc)
                        item['pred_rebel']  = get_winner(reb)
                        archive_data['pred_anchor'] = item['pred_anchor']
                        archive_data['pred_rebel']  = item['pred_rebel']

                        imp_h  = 1 / h_odd
                        imp_a  = 1 / a_odd
                        edge_h = reb['home'] - imp_h
                        edge_a = reb['away'] - imp_a

                        if edge_h > 0.05:
                            item['verdict']       = f"HOME VALUE (+{edge_h*100:.1f}%)"
                            item['verdict_class'] = 'value-home'
                        elif edge_a > 0.05:
                            item['verdict']       = f"AWAY VALUE (+{edge_a*100:.1f}%)"
                            item['verdict_class'] = 'value-away'
                        elif reb['draw'] > 0.29:
                            item['verdict']       = 'DRAW WATCH'
                            item['verdict_class'] = 'value-draw'

                    unique_id = f"{league_code}-{home}-{away}"
                    self._save_to_archive(unique_id, archive_data)

                except Exception as e:
                    print(f"Prediction error ({home} vs {away}): {e}")
                    item['verdict'] = 'Data Error'

                live_games.append(item)

        history = self._get_full_history(league_code)
        self._update_accuracy_stats(history, league_code)

        return {'live': live_games, 'history': history, 'league_name': config['name']}

    # -------------------------------------------------------------------------
    # ODDS API
    # -------------------------------------------------------------------------
    def _fetch_odds_api(self, sport, region):
        url    = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
        params = {'api_key': API_KEY, 'regions': region, 'markets': MARKETS, 'oddsFormat': 'decimal'}
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            print(f"Odds API {resp.status_code}: {resp.text[:200]}")
            return []
        except Exception as e:
            print(f"Odds API error: {e}")
            return []

    # -------------------------------------------------------------------------
    # HISTORY
    # -------------------------------------------------------------------------
    def _get_full_history(self, league_code):
        """
        Build history list. code_2.get_history already computes anchor/rebel
        predictions for each historical game — use those directly as fallback.
        Trinity is only available from the Supabase archive (written at live
        prediction time). The archive key matches _save_to_archive.
        """
        try:
            raw = code_2.get_history_data(league_code=league_code)
            clean = []
            for h in raw:
                home = self._normalize_name(h['home_team'])
                away = self._normalize_name(h['away_team'])
                uid  = f"{league_code}-{home}-{away}"
                saved = self.archive.get(uid, {})
                clean.append({
                    'date':         h['date'],
                    'home_team':    home,
                    'away_team':    away,
                    'fixture':      h['fixture'],
                    'result':       h['result'],
                    # Archive predictions are locked at live-prediction time.
                    # For anchor/rebel, fall back to code_2 re-computed values
                    # so history rows always have something to score against.
                    'pred_trinity': saved.get('pred_trinity') or 'N/A',
                    'pred_anchor':  saved.get('pred_anchor')  or h.get('pred_anchor', 'N/A'),
                    'pred_rebel':   saved.get('pred_rebel')   or h.get('pred_rebel', 'N/A'),
                })
            return clean
        except Exception as e:
            print(f"History error ({league_code}): {e}")
            return []

    # -------------------------------------------------------------------------
    # ACCURACY STATS
    # -------------------------------------------------------------------------
    def _update_accuracy_stats(self, history_games, league_code):
        for game in history_games:
            game_id = f"{league_code}-{game['date']}-{game['fixture']}"
            if '?' in game['result']:
                continue
            if self._is_processed(game_id):
                continue
            try:
                res_char = game['result'].split('(')[1].replace(')', '')
                actual = 'HOME' if res_char == 'H' else ('AWAY' if res_char == 'A' else 'DRAW')
            except Exception:
                continue
            for model in ['trinity', 'anchor', 'rebel']:
                pred = game.get(f'pred_{model}', 'N/A')
                if pred == 'N/A':
                    continue
                self._save_stat(league_code, model, 1 if pred == actual else 0, 1)
            self._mark_processed(game_id)

    # -------------------------------------------------------------------------
    # PUBLIC HELPERS (used by app.py)
    # -------------------------------------------------------------------------
    def get_stats(self):
        return self._load_stats()

    def get_model_metrics(self):
        """
        Returns per-league model quality metrics (log-loss, F1) from the
        most recent training run. Used by /the-ai to display calibration quality.
        Structure: { league_code: { 'trinity': {log_loss, log_loss_std},
                                    'anchor':  {f1, log_loss},
                                    'rebel':   {f1, log_loss} } }
        """
        result = {}
        for league_code in LEAGUE_CONFIG:
            entry = {}
            # Trinity metrics (stored on the engine object in code_1._engines)
            try:
                from models import code_1
                engine = code_1._engines.get(league_code)
                if engine:
                    entry['trinity'] = getattr(engine, 'cv_metrics', {})
            except Exception:
                pass
            # Anchor/Rebel metrics (stored on HybridPipeline in code_2._pipelines)
            try:
                entry.update(code_2.get_model_metrics(league_code))
            except Exception:
                pass
            result[league_code] = entry
        return result

    def retrain_all(self):
        """
        4.3: Re-run full pipeline for all five leagues — data fetch, feature
        engineering, and model training. Called after /refresh-global so models
        always reflect the latest completed matchday results.
        Runs synchronously inside the background thread started by /refresh-global.
        """
        print(":: RETRAIN START ::")
        for league_code in LEAGUE_CONFIG:
            try:
                print(f"  Retraining {league_code}...")
                # Calling _generate_fresh_data with cleared cache forces full
                # re-fetch + re-engineering + re-training via code_1/code_2
                self._generate_fresh_data(league_code)
                self._set_cache(league_code, self._generate_fresh_data(league_code))
                print(f"  {league_code} retrained and cached.")
            except Exception as e:
                print(f"  Retrain error ({league_code}): {e}")
        print(":: RETRAIN COMPLETE ::")

    def get_last_updated(self):
        if not supabase:
            return 'Unknown'
        try:
            res = supabase.table('model_stats').select('updated_at').order(
                'updated_at', desc=True
            ).limit(1).execute()
            if res.data:
                raw = res.data[0]['updated_at']
                dt  = datetime.fromisoformat(raw.replace('Z', '+00:00'))
                return dt.strftime('%d %b %Y @ %H:%M')
        except Exception as e:
            print(f"Last updated error: {e}")
        return 'Unknown'
