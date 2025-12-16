import sys
import os
import time

# Ensure we can find the models folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from model_manager import ModelManager
except ImportError:
    print("!! CRITICAL ERROR: Could not import ModelManager from models/model_manager.py")
    sys.exit(1)

# --- CONFIG ---
LEAGUES_TO_UPDATE = ['E0', 'SP1', 'D1', 'I1', 'F1']

def force_update_all():
    """
    1. Clears existing cache (to force fresh API calls).
    2. Iterates through all leagues.
    3. Triggers ModelManager to fetch fresh odds & recalculate stats.
    4. Returns True if successful.
    """
    print("\n=== STARTING GLOBAL AI REFRESH ===")
    start_time = time.time()
    
    # Initialize your existing manager
    manager = ModelManager()
    
    # 1. NUKE THE CACHE
    # We want fresh data, not the 10-day old saved files.
    print(":: Clearing old cache files...")
    manager.clear_cache()
    
    # 2. UPDATE EACH LEAGUE
    # Calling get_dashboard_data() triggers the whole chain in your manager:
    # Fetch API -> Run code_1/code_2 -> Update History -> Update Stats JSON
    for code in LEAGUES_TO_UPDATE:
        print(f"   > Updating {code} (Fetching Odds & Running Models)...")
        try:
            # This function internally updates 'model_stats.json'
            manager.get_dashboard_data(league_code=code)
            print(f"     [✓] {code} Complete.")
        except Exception as e:
            print(f"     [X] Failed {code}: {e}")
            
    elapsed = time.time() - start_time
    print(f"=== REFRESH COMPLETE in {elapsed:.1f}s ===\n")
    return True

if __name__ == "__main__":
    force_update_all()