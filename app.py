import os
import math
import threading
import traceback
from flask import Flask, render_template, redirect, url_for, request
from model_manager import ModelManager
import master_feed

app = Flask(__name__)
manager = ModelManager()

# Pre-warm all league engines in a background thread so the first real
# request doesn't have to wait 60-90 seconds for training to complete.
def _background_warmup():
    manager.warm_up()

_warmup_thread = threading.Thread(target=_background_warmup, daemon=True)
_warmup_thread.start()

# --- CONFIGURATION ---
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    print("WARNING: ADMIN_PASSWORD not set in Render Environment Variables")

LEAGUE_MAP = {
    'E0':  'Premier League (UK)',
    'SP1': 'La Liga (Spain)',
    'D1':  'Bundesliga (Germany)',
    'I1':  'Serie A (Italy)',
    'F1':  'Ligue 1 (France)'
}

# Tracks whether a global refresh is running (prevents double-triggering)
_refresh_running = False


# --- HELPER: Stats for /the-ai page ---
def calculate_live_stats():
    """
    Pulls accuracy stats from Supabase (via manager) and formats them
    for the /the-ai template. Derives percentage from correct/total counts.
    """
    raw = manager.get_stats()
    stats_output = {}

    for code, name in LEAGUE_MAP.items():
        league_stats = {
            "name":    name,
            "trinity": {"cum": "0%"},
            "anchor":  {"cum": "0%"},
            "rebel":   {"cum": "0%"},
        }
        if code in raw:
            for model in ['trinity', 'anchor', 'rebel']:
                d       = raw[code].get(model, {})
                correct = d.get('correct', 0)
                total   = d.get('total', 0)
                if total > 0:
                    league_stats[model]['cum'] = f"{int((correct / total) * 100)}%"
        stats_output[code] = league_stats

    return stats_output


# --- ROUTES ---

@app.route('/')
def home():
    league_code = request.args.get('league')

    if not league_code:
        return render_template('index.html', leagues=LEAGUE_MAP)

    if league_code not in LEAGUE_MAP:
        league_code = 'E0'

    # FIX 2.4: Show warmup page while engines are still loading
    if not manager._is_warm:
        return """
        <html>
        <head>
            <meta http-equiv="refresh" content="10">
            <style>
                body {{ background:#0b0c10; color:#66fcf1; font-family:monospace;
                        display:flex; align-items:center; justify-content:center;
                        height:100vh; margin:0; text-align:center; }}
                .dot {{ animation: blink 1.2s infinite; }}
                @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.2}} }}
            </style>
        </head>
        <body>
            <div>
                <h2>:: SYSTEM WARMING UP <span class="dot">...</span> ::</h2>
                <p style="color:#888">AI models are loading. This page will refresh automatically every 10 seconds.</p>
                <p style="font-size:0.8rem; color:#555">First boot takes 60-90 seconds. Subsequent loads use the cache.</p>
            </div>
        </body>
        </html>
        """

    try:
        data = manager.get_dashboard_data(league_code=league_code)
        data['league_name'] = LEAGUE_MAP[league_code]
        return render_template('dashboard.html', data=data, league_code=league_code)
    except Exception as e:
        print(f"CRITICAL ERROR in Dashboard for {league_code}: {e}")
        traceback.print_exc()
        return f"<h3>System Error loading {league_code}</h3><p>Check server logs.</p>", 500


@app.route('/refresh')
def refresh():
    pwd           = request.args.get('pwd')
    target_league = request.args.get('league')

    if pwd == ADMIN_PASSWORD:
        manager.clear_cache()
        if target_league and target_league in LEAGUE_MAP:
            return redirect(url_for('home', league=target_league))
        return redirect(url_for('home'))
    return "<h3>ACCESS DENIED: Authorization Required.</h3>", 403


@app.route('/refresh-global')
def refresh_global():
    """
    Triggers a full AI refresh in a background thread so the HTTP
    request returns immediately instead of timing out after 30s.
    """
    global _refresh_running

    pwd = request.args.get('pwd')
    if pwd != ADMIN_PASSWORD:
        return "<h3>ACCESS DENIED. AUTHORIZATION REQUIRED.</h3>", 403

    if _refresh_running:
        return """
        <h3>Refresh already in progress.</h3>
        <p>The AI is currently updating. Check back in a few minutes.</p>
        <a href='/the-ai'>Return to The AI</a>
        """

    def run_refresh():
        global _refresh_running
        _refresh_running = True
        try:
            print(":: GLOBAL AI REFRESH STARTED (background thread) ::")
            master_feed.force_update_all()
            manager.clear_cache()

            # 4.3: Retrain all five league models after new results are in.
            # Pipeline cache is already cleared above, so next get_dashboard_data
            # call will re-run full data fetch + feature engineering + training.
            # We trigger that explicitly here so users see fresh models immediately.
            print(":: RETRAINING ALL LEAGUE MODELS ::")
            manager.retrain_all()
            print(":: GLOBAL AI REFRESH + RETRAIN COMPLETE ::")
        except Exception as e:
            print(f"Global Refresh Error: {e}")
        finally:
            _refresh_running = False

    thread = threading.Thread(target=run_refresh, daemon=True)
    thread.start()

    # Return immediately — don't wait for the thread
    next_page   = request.args.get('next')
    league_code = request.args.get('league')

    return """
    <html>
    <head>
        <meta http-equiv="refresh" content="5;url={redirect_url}">
        <style>
            body {{ background:#0b0c10; color:#66fcf1; font-family:monospace;
                    display:flex; align-items:center; justify-content:center;
                    height:100vh; margin:0; text-align:center; }}
        </style>
    </head>
    <body>
        <div>
            <h2>:: AI REFRESH INITIATED ::</h2>
            <p>Running in background. Redirecting in 5 seconds...</p>
            <p style="color:#888; font-size:0.8rem">This process takes 3-5 minutes.
            Come back to /the-ai to see updated stats.</p>
        </div>
    </body>
    </html>
    """.format(
        redirect_url=url_for('home', league=league_code)
        if next_page == 'dashboard' and league_code
        else url_for('the_ai')
    )


@app.route('/the-ai')
def the_ai():
    live_stats       = calculate_live_stats()
    model_metrics    = manager.get_model_metrics()   # 4.1: calibration quality
    last_update_time = manager.get_last_updated()

    # FIX 1.4: Calculate matchday from whichever league data is already warm,
    # not hardcoded to E0 — avoids gating all five leagues on E0 game count.
    current_matchday = 1
    for lc, cfg in {'E0': 10, 'SP1': 10, 'D1': 9, 'I1': 10, 'F1': 10}.items():
        try:
            if _memory_cache.get(lc):  # only use already-cached leagues
                from model_manager import _memory_cache
                cached_data = _memory_cache[lc]['data']
                games_played = len(cached_data.get('history', []))
                games_per_round = cfg
                current_matchday = max(current_matchday, math.ceil(games_played / games_per_round) + 1)
        except Exception:
            pass

    return render_template(
        'the_ai.html',
        stats          = live_stats,
        model_metrics  = model_metrics,
        last_updated   = last_update_time,
        matchday       = current_matchday,
        league_code    = None,
    )


if __name__ == '__main__':
    app.run(debug=True)
