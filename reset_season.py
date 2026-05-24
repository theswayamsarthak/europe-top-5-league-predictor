"""
reset_season.py
===============
End-of-season reset script.

Run ONCE after the last matchday of each season to:
  1. Print a final accuracy summary for all 3 models across all 5 leagues.
  2. Clear cumulative accuracy stats (model_stats table in Supabase OR local JSON).
  3. Clear the processed_results table so next season's games are scored fresh.
  4. Optionally archive the predictions archive for the completed season.

Usage:
  source .env && python reset_season.py

Add --dry-run to preview without making any changes.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="End-of-season reset for the predictor.")
parser.add_argument('--dry-run', action='store_true', help='Preview actions without changing anything.')
parser.add_argument('--season', default='2025-26', help='Label for the season being archived (default: 2025-26).')
args = parser.parse_args()

DRY = args.dry_run
SEASON = args.season

print(f"\n{'[DRY RUN] ' if DRY else ''}End-of-Season Reset — {SEASON}")
print("=" * 60)

# ── Supabase setup ────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Supabase connected.\n")
    except Exception as e:
        print(f"✗ Supabase connection failed: {e}\n")
else:
    print("⚠  No Supabase credentials — operating on local files only.\n")

LEAGUES = ['E0', 'SP1', 'D1', 'I1', 'F1']
LEAGUE_NAMES = {
    'E0': 'Premier League', 'SP1': 'La Liga',
    'D1': 'Bundesliga', 'I1': 'Serie A', 'F1': 'Ligue 1'
}
MODELS = ['trinity', 'anchor', 'rebel']

# ── Step 1: Print final accuracy summary ─────────────────────────────────────
print("STEP 1 — Final Season Accuracy Summary")
print("-" * 60)

stats = {}

if supabase:
    try:
        result = supabase.table('model_stats').select('*').execute()
        for row in result.data:
            league = row['league']
            model = row['model']
            stats.setdefault(league, {})[model] = {
                'correct': row.get('correct', 0),
                'total': row.get('total', 0),
            }
    except Exception as e:
        print(f"Could not fetch Supabase stats: {e}")

for league in LEAGUES:
    print(f"\n  {LEAGUE_NAMES.get(league, league)}:")
    for model in MODELS:
        d = stats.get(league, {}).get(model, {})
        correct = d.get('correct', 0)
        total = d.get('total', 0)
        if total > 0:
            pct = int((correct / total) * 100)
            print(f"    {model.capitalize():8s}: {correct}/{total} = {pct}%")
        else:
            print(f"    {model.capitalize():8s}: no data")

print()

# ── Step 2: Archive the season stats snapshot ─────────────────────────────────
print("STEP 2 — Archiving stats snapshot")
print("-" * 60)

archive_filename = f"season_stats_{SEASON.replace('/', '-')}.json"
archive_data = {
    'season': SEASON,
    'archived_at': datetime.utcnow().isoformat(),
    'stats': stats,
}

if not DRY:
    with open(archive_filename, 'w') as f:
        json.dump(archive_data, f, indent=2)
    print(f"  ✓ Saved to {archive_filename}")
else:
    print(f"  [DRY] Would save to {archive_filename}")

# ── Step 3: Clear Supabase model_stats ────────────────────────────────────────
print("\nSTEP 3 — Resetting cumulative accuracy stats")
print("-" * 60)

if supabase:
    for league in LEAGUES:
        for model in MODELS:
            stat_id = f"{league}_{model}"
            if not DRY:
                try:
                    supabase.table('model_stats').upsert({
                        'stat_id': stat_id,
                        'league': league,
                        'model': model,
                        'correct': 0,
                        'total': 0,
                        'updated_at': datetime.utcnow().isoformat(),
                    }).execute()
                    print(f"  ✓ Reset {stat_id}")
                except Exception as e:
                    print(f"  ✗ Failed to reset {stat_id}: {e}")
            else:
                print(f"  [DRY] Would reset {stat_id}")
else:
    print("  ⚠  No Supabase — stats reset only applies when Supabase is configured.")
    print("     Cumulative stats are not persisted locally between restarts, so nothing to clear.")

# ── Step 4: Clear processed_results ──────────────────────────────────────────
print("\nSTEP 4 — Clearing processed results (prevents double-scoring next season)")
print("-" * 60)

if supabase:
    if not DRY:
        try:
            # Delete all rows (Supabase requires a filter; use a truthy condition on the pk column)
            supabase.table('processed_results').delete().neq('game_id', '___never___').execute()
            print("  ✓ processed_results table cleared.")
        except Exception as e:
            print(f"  ✗ Failed to clear processed_results: {e}")
    else:
        print("  [DRY] Would delete all rows from processed_results.")
else:
    print("  ⚠  No Supabase — nothing to clear.")

# ── Step 5: Clear predictions archive (optional) ─────────────────────────────
print("\nSTEP 5 — Clearing predictions archive for new season")
print("-" * 60)

LOCAL_ARCHIVE = 'predictions_archive.json'
if supabase:
    if not DRY:
        try:
            supabase.table('predictions_archive').delete().neq('game_id', '___never___').execute()
            print("  ✓ Supabase predictions_archive cleared.")
        except Exception as e:
            print(f"  ✗ Failed to clear Supabase archive: {e}")
    else:
        print("  [DRY] Would delete all rows from predictions_archive.")

if os.path.exists(LOCAL_ARCHIVE):
    if not DRY:
        backup_name = f"predictions_archive_{SEASON.replace('/', '-')}.json"
        os.rename(LOCAL_ARCHIVE, backup_name)
        with open(LOCAL_ARCHIVE, 'w') as f:
            json.dump({}, f)
        print(f"  ✓ Local archive backed up to {backup_name} and reset.")
    else:
        print(f"  [DRY] Would back up {LOCAL_ARCHIVE} and reset it to {{}}.")

# ── Done ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if DRY:
    print("DRY RUN complete — no changes made.")
    print("Remove --dry-run to apply the reset.")
else:
    print(f"Season {SEASON} reset complete.")
    print("You're ready for the new season. Remember to:")
    print("  • Update CURRENT_SEASON in injuries_engine.py (line ~37)")
    print("  • The season codes in code_1.py and code_2.py auto-extend — no change needed.")
print()
