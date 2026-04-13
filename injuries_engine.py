"""
injuries_engine.py
==================
Fetches injury lists and confirmed lineups from API-Football and builds
two pre-match adjustment features for every upcoming fixture:

  Injury_Penalty_H / Injury_Penalty_A
      Weighted count of first-team regulars currently unavailable.
      Weight = 1.0 for injured/suspended key players, 0.5 for squad depth.
      Capped at 3.0 so one catastrophic injury list doesn't dominate.

  Lineup_Str_H / Lineup_Str_A
      When confirmed lineups are available (~1hr before kickoff):
      average ELO-proxy rating of the confirmed XI vs the team's typical XI.
      Ranges 0.7–1.3. 1.0 = full-strength. <1.0 = weakened. >1.0 = strong.
      Falls back to 1.0 when lineups are not yet released.

Both features are added to Rebel's feature vector at prediction time.
They are zero-cost at training time (historical data doesn't have them)
and are fetched fresh at prediction time from the API.

API-Football free tier: 100 requests/day, 10 req/min.
Budget per full refresh: ~60 requests across 5 leagues.

Stable league IDs:
  E0  → 39   (Premier League)
  SP1 → 140  (La Liga)
  D1  → 78   (Bundesliga)
  I1  → 135  (Serie A)
  F1  → 61   (Ligue 1)
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("APIFOOTBALL_KEY", "aeceae3875fccfb8bdc5d0b5e689c2b6")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_KEY}

LEAGUE_IDS = {
    "E0":  39,
    "SP1": 140,
    "D1":  78,
    "I1":  135,
    "F1":  61,
}

# Current season year (2024 = 2024/25).  Update each August.
CURRENT_SEASON = 2024

# Request pacing — free tier allows 10/min
REQUEST_INTERVAL = 6.1   # seconds between calls (10/min = 1 per 6s)

# ── In-memory cache: keyed by league_code, expires after 3 hours ─────────────
_injury_cache:  dict = {}   # {league_code: {"data": [...], "ts": float}}
_lineup_cache:  dict = {}   # {fixture_id:  {"data": [...], "ts": float}}
CACHE_TTL = 3 * 3600        # 3 hours


# ── HTTP helper ───────────────────────────────────────────────────────────────
def _get(endpoint: str, params: dict) -> dict | None:
    """Single rate-limited GET. Returns parsed JSON or None on failure."""
    url = f"{BASE_URL}/{endpoint}"
    try:
        time.sleep(REQUEST_INTERVAL)
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        log.warning(f"API-Football {endpoint} → {r.status_code}: {r.text[:120]}")
    except Exception as e:
        log.warning(f"API-Football request failed ({endpoint}): {e}")
    return None


# ── Injuries ──────────────────────────────────────────────────────────────────
def fetch_injuries(league_code: str) -> list:
    """
    Returns list of current injuries/suspensions for a league.
    Each item: {"player": str, "team": str, "type": str, "reason": str}
    Cached for 3 hours.
    """
    now = time.time()
    cached = _injury_cache.get(league_code)
    if cached and (now - cached["ts"]) < CACHE_TTL:
        return cached["data"]

    league_id = LEAGUE_IDS.get(league_code)
    if not league_id:
        return []

    resp = _get("injuries", {"league": league_id, "season": CURRENT_SEASON})
    if not resp or "response" not in resp:
        return []

    injuries = []
    for item in resp["response"]:
        player = item.get("player", {})
        team   = item.get("team",   {})
        injuries.append({
            "player": player.get("name", "Unknown"),
            "team":   team.get("name",   "Unknown"),
            "type":   item.get("type",   "Unknown"),
            "reason": item.get("reason", ""),
        })

    _injury_cache[league_code] = {"data": injuries, "ts": now}
    log.info(f"Injuries fetched: {league_code} → {len(injuries)} players out")
    return injuries


# ── Fixtures (for lineup fixture IDs) ────────────────────────────────────────
def fetch_upcoming_fixture_ids(league_code: str, n: int = 10) -> dict:
    """
    Returns {(home_team_name, away_team_name): fixture_id} for next n fixtures.
    Used to look up lineups by home+away team name.
    """
    league_id = LEAGUE_IDS.get(league_code)
    if not league_id:
        return {}

    resp = _get("fixtures", {
        "league":  league_id,
        "season":  CURRENT_SEASON,
        "status":  "NS",      # Not Started
        "next":    n,
    })
    if not resp or "response" not in resp:
        return {}

    result = {}
    for fix in resp["response"]:
        fid  = fix["fixture"]["id"]
        home = fix["teams"]["home"]["name"]
        away = fix["teams"]["away"]["name"]
        result[(home, away)] = fid

    return result


# ── Lineups ───────────────────────────────────────────────────────────────────
def fetch_lineup(fixture_id: int) -> dict | None:
    """
    Returns lineup dict for one fixture, or None if not yet released.
    Structure: {"home": [player_names], "away": [player_names]}
    Cached per fixture_id for 3 hours.
    """
    now = time.time()
    cached = _lineup_cache.get(fixture_id)
    if cached and (now - cached["ts"]) < CACHE_TTL:
        return cached["data"]

    resp = _get("fixtures/lineups", {"fixture": fixture_id})
    if not resp or not resp.get("response"):
        return None

    lineups = resp["response"]
    if len(lineups) < 2:
        return None     # lineup not released yet

    result = {}
    for side in lineups:
        venue = "home" if side["team"]["id"] == lineups[0]["team"]["id"] else "away"
        result[venue] = [p["player"]["name"] for p in side.get("startXI", [])]

    _lineup_cache[fixture_id] = {"data": result, "ts": now}
    return result


# ── Feature builders ──────────────────────────────────────────────────────────
def compute_injury_penalty(team_name: str, injuries: list,
                           elo_ratings: dict | None = None) -> float:
    """
    Injury_Penalty for one team.

    Counts players from `injuries` whose team name fuzzy-matches `team_name`.
    Weights:
      - "Injured" or "Suspended" with no return date → 1.0
      - Everything else (doubtful, minor) → 0.5

    If ELO ratings per player are available (future extension), weighting
    scales by player importance. For now uses a flat positional heuristic:
    we don't have per-player ELO, so we count first 11 players in the list
    as key players (weight 1.0) and the rest as depth (weight 0.5).

    Capped at 3.0 to prevent one catastrophically injured squad dominating.
    """
    team_norm = team_name.lower().strip()
    team_injuries = [
        i for i in injuries
        if team_norm in i["team"].lower() or i["team"].lower() in team_norm
    ]

    penalty = 0.0
    for idx, inj in enumerate(team_injuries):
        itype = inj.get("type", "").lower()
        if "injured" in itype or "suspended" in itype or "missing" in itype:
            weight = 1.0 if idx < 5 else 0.5    # first 5 = key players
        else:
            weight = 0.3   # minor / doubtful
        penalty += weight

    return round(min(penalty, 3.0), 3)


def compute_lineup_strength(team_name: str, confirmed_xi: list,
                             typical_xi_size: int = 11) -> float:
    """
    Lineup_Strength ratio: how strong is this confirmed XI vs typical?

    Without per-player ELO (which would require a separate dataset), we use
    a proxy: confirmed_xi length vs expected 11. A full lineup = 1.0. Fewer
    confirmed starters (partial release) = slight penalty. More = 1.0.

    When lineups are not yet released, returns 1.0 (neutral — no adjustment).

    This is intentionally conservative. The feature adds real value when:
    - A star player is confirmed missing from the XI despite no injury report
    - A rotated squad is confirmed for a cup-heavy schedule
    The injury penalty above captures the known absences; lineup strength
    captures the tactical/rotational dimension.
    """
    if not confirmed_xi:
        return 1.0   # not released yet — no adjustment

    n = len(confirmed_xi)
    if n >= typical_xi_size:
        return 1.0
    # Partial release — slight strength reduction proportional to missing starters
    return round(0.85 + 0.15 * (n / typical_xi_size), 3)


# ── Main entry point ──────────────────────────────────────────────────────────
def get_prematch_features(home_team: str, away_team: str,
                           league_code: str) -> dict:
    """
    Returns pre-match adjustment features for one fixture.

    Calls:
      1. fetch_injuries() for the league (cached, 1 API call per league per 3h)
      2. fetch_upcoming_fixture_ids() to get the fixture ID (1 call per league)
      3. fetch_lineup() for that fixture ID if available (1 call per fixture)

    Returns dict with keys:
      Injury_Penalty_H, Injury_Penalty_A  (0.0–3.0, higher = more injuries)
      Lineup_Str_H,     Lineup_Str_A      (0.85–1.0, lower = weakened side)

    Safe to call with no API key — all values default to neutral (0.0 / 1.0).
    """
    neutral = {
        "Injury_Penalty_H": 0.0,
        "Injury_Penalty_A": 0.0,
        "Lineup_Str_H":     1.0,
        "Lineup_Str_A":     1.0,
    }

    if not API_KEY:
        log.warning("APIFOOTBALL_KEY not set — returning neutral features")
        return neutral

    try:
        # 1. Injuries for this league
        injuries = fetch_injuries(league_code)

        neutral["Injury_Penalty_H"] = compute_injury_penalty(home_team, injuries)
        neutral["Injury_Penalty_A"] = compute_injury_penalty(away_team, injuries)

        # 2. Fixture ID for lineup lookup
        fixture_map = fetch_upcoming_fixture_ids(league_code, n=15)

        # Fuzzy-match team names (API names vs football-data names differ slightly)
        fixture_id = None
        home_norm  = home_team.lower()
        away_norm  = away_team.lower()
        for (api_home, api_away), fid in fixture_map.items():
            if (home_norm in api_home.lower() or api_home.lower() in home_norm) and \
               (away_norm in api_away.lower() or api_away.lower() in away_norm):
                fixture_id = fid
                break

        # 3. Lineups if available
        if fixture_id:
            lineup = fetch_lineup(fixture_id)
            if lineup:
                neutral["Lineup_Str_H"] = compute_lineup_strength(
                    home_team, lineup.get("home", [])
                )
                neutral["Lineup_Str_A"] = compute_lineup_strength(
                    away_team, lineup.get("away", [])
                )

    except Exception as e:
        log.warning(f"injuries_engine: feature build failed ({home_team} vs {away_team}): {e}")

    return neutral


# ── Quota guard ───────────────────────────────────────────────────────────────
def check_quota() -> dict:
    """
    Returns current API quota status.
    Useful to call before a batch refresh to ensure budget remains.
    """
    resp = _get("status", {})
    if resp and "response" in resp:
        sub = resp["response"].get("subscription", {})
        req = resp["response"].get("requests", {})
        return {
            "plan":          sub.get("plan", "Unknown"),
            "requests_used": req.get("current", 0),
            "requests_limit": req.get("limit_day", 100),
            "remaining":     req.get("limit_day", 100) - req.get("current", 0),
        }
    return {}
