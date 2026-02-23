"""
=============================================================================
master.py — Human Control Center for Territorial.io AI
=============================================================================

The central command interface for managing the AI training system.
Run this on YOUR computer (not Kaggle/Colab) to:

1. Review moves — Launch the terminal review interface
2. Merge sessions — Combine all 6 session files into master_model
3. Update best moves — Extract top-rated moves from rated_moves.json  
4. Show dashboard — View status of all 6 sessions at a glance
5. Health check — Verify all files exist and are valid
6. Pull all sessions — Download all 6 session files from GitHub
7. Exit

Usage:
    python master.py

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

import os
import sys
import json
import time
import importlib
from typing import Dict, List


# ==============================================
# CONFIGURATION
# ==============================================

SESSION_NAMES = ["colab1", "colab2", "colab3", "colab4", "kaggle1", "kaggle2"]

ALL_FILES = {
    "python": [
        "brain.py", "decision.py", "memory.py", "rewards.py",
        "curriculum.py", "trainer.py", "game_environment.py",
        "screen_capture.py", "action_controller.py", "move_recorder.py",
        "merger.py", "sync.py", "review_interface.py",
        "run_session.py", "master.py",
    ],
    "json": [
        "config.json", "review_queue.json", "rated_moves.json",
        "master_model.json", "best_moves.json",
    ],
    "session_json": [f"session_{name}.json" for name in SESSION_NAMES],
}


# ==============================================
# MENU FUNCTIONS
# ==============================================

def review_moves():
    """Launch the terminal review interface."""
    print("\n" + "=" * 60)
    print("📋 LAUNCHING MOVE REVIEW INTERFACE")
    print("=" * 60)

    try:
        from review_interface import run_review_session
        run_review_session()
    except ImportError as e:
        print(f"❌ Could not import review_interface: {e}")
    except Exception as e:
        print(f"❌ Error during review: {e}")


def merge_sessions():
    """Run the session merger to combine all 6 sessions."""
    print("\n" + "=" * 60)
    print("🔄 MERGING ALL SESSIONS")
    print("=" * 60)

    try:
        from merger import SessionMerger
        merger = SessionMerger()
        result = merger.merge_all()
        merger.print_merge_report()

        # Auto-push master_model.json to GitHub
        try:
            from sync import GitHubSync
            sync = GitHubSync()
            print("\n🔄 Pushing merged master model to GitHub...")
            sync._retry(lambda: sync._push_file("master_model.json", "master_model.json"))
            print("✅ Master model pushed!")
        except ImportError:
            print("⚠️ sync.py not available — skip GitHub push")
        except Exception as e:
            print(f"⚠️ Could not push to GitHub: {e}")

    except ImportError as e:
        print(f"❌ Could not import merger: {e}")
    except Exception as e:
        print(f"❌ Error during merge: {e}")


def update_best_moves():
    """Extract top-rated moves from rated_moves.json and update best_moves.json."""
    print("\n" + "=" * 60)
    print("⭐ UPDATING BEST MOVES")
    print("=" * 60)

    rated_path = "rated_moves.json"
    best_path = "best_moves.json"

    if not os.path.exists(rated_path):
        print("⚠️ No rated_moves.json found. Rate some moves first!")
        return

    try:
        with open(rated_path, 'r') as f:
            rated = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ Error reading rated_moves.json: {e}")
        return

    best = []
    for move in rated:
        ratings = move.get("human_ratings", {})
        if not ratings:
            continue

        good_count = sum(1 for r in ratings.values() if r == "good")
        if good_count >= 7:
            best_entry = {
                "move_id": move.get("move_id", "unknown"),
                "session": move.get("session", "unknown"),
                "game_number": move.get("game_number", 0),
                "action_type": move.get("action_type", "unknown"),
                "game_situation": f"Game {move.get('game_number', 0)}, "
                                 f"move {move.get('move_number', 0)}",
                "all_10_factor_ratings": ratings,
                "why_great": f"Rated 'good' on {good_count}/10 factors by human",
                "how_to_replicate": f"Action: {move.get('action_type', 'unknown')} "
                                    f"at grid ({move.get('grid_row', '?')}, "
                                    f"{move.get('grid_col', '?')})",
            }
            best.append(best_entry)

    with open(best_path, 'w') as f:
        json.dump(best, f, indent=2)

    print(f"✅ Updated {best_path}: {len(best)} best moves "
          f"(from {len(rated)} rated)")

    # Auto-push to GitHub
    try:
        from sync import GitHubSync
        sync = GitHubSync()
        sync._retry(lambda: sync._push_file(best_path, best_path))
    except Exception:
        pass


def show_dashboard():
    """Print a dashboard showing status of all 6 sessions."""
    print("\n" + "=" * 70)
    print("📊 TRAINING DASHBOARD")
    print("=" * 70)

    print(f"\n  {'Session':<12} {'Games':>6} {'Win%':>6} {'Phase':>25} "
          f"{'Difficulty':>10} {'Queue':>6} {'Last Active':>16}")
    print("  " + "-" * 84)

    for name in SESSION_NAMES:
        session_file = f"session_{name}.json"
        if not os.path.exists(session_file):
            # Check sessions directory
            session_file = os.path.join("sessions", f"session_{name}.json")

        if os.path.exists(session_file):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)

                games = _extract(data, "total_games_played", "games_played", 0)
                win_rate = _extract(data, "win_rate", None, 0.0)
                phase = _extract(data, "current_phase", "phase", "unknown")
                difficulty = _extract(data, "current_difficulty", "difficulty", "unknown")
                last_updated = _extract(data, "last_updated", None, 0)

                # Count queue items for this session
                queue_count = _count_queue_items(name)

                # Format last active
                if last_updated > 0:
                    last_active = time.strftime("%m/%d %H:%M", time.localtime(last_updated))
                else:
                    last_active = "unknown"

                print(f"  {name:<12} {games:>6} {win_rate:>5.0%} "
                      f"{phase:>25} {difficulty:>10} {queue_count:>6} {last_active:>16}")

            except Exception as e:
                print(f"  {name:<12} {'ERROR':>6} — {str(e)[:50]}")
        else:
            print(f"  {name:<12} {'—':>6} {'—':>6} {'no data':>25} {'—':>10} {'—':>6} {'—':>16}")

    print()

    # Overall stats
    total_queue = _count_queue_items()
    rated_count = 0
    if os.path.exists("rated_moves.json"):
        try:
            with open("rated_moves.json", 'r') as f:
                rated_count = len(json.load(f))
        except Exception:
            pass

    best_count = 0
    if os.path.exists("best_moves.json"):
        try:
            with open("best_moves.json", 'r') as f:
                best_count = len(json.load(f))
        except Exception:
            pass

    print(f"  📋 Review Queue: {total_queue} moves waiting")
    print(f"  ✅ Rated Moves: {rated_count}")
    print(f"  ⭐ Best Moves: {best_count}")

    print()


def health_check():
    """Verify all files exist, are valid, and import successfully."""
    print("\n" + "=" * 60)
    print("🏥 HEALTH CHECK")
    print("=" * 60)

    issues = 0

    # Check Python files
    print("\n  🐍 Python Files:")
    for py_file in ALL_FILES["python"]:
        if os.path.exists(py_file):
            # Try to import
            module_name = py_file.replace(".py", "")
            try:
                mod = importlib.import_module(module_name)
                print(f"     ✅ {py_file}")
            except Exception as e:
                print(f"     ⚠️ {py_file} — import error: {str(e)[:60]}")
                issues += 1
        else:
            print(f"     ❌ {py_file} — MISSING")
            issues += 1

    # Check JSON files
    print("\n  📄 JSON Files:")
    for json_file in ALL_FILES["json"]:
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    json.load(f)
                print(f"     ✅ {json_file}")
            except json.JSONDecodeError as e:
                print(f"     ⚠️ {json_file} — invalid JSON: {e}")
                issues += 1
        else:
            print(f"     ❌ {json_file} — MISSING (will be created during training)")

    # Check session files
    print("\n  💻 Session Files:")
    for session_file in ALL_FILES["session_json"]:
        found = os.path.exists(session_file) or os.path.exists(
            os.path.join("sessions", session_file))
        if found:
            print(f"     ✅ {session_file}")
        else:
            print(f"     ⚠️ {session_file} — not yet created")

    # Check directories
    print("\n  📁 Directories:")
    for dir_name in ["sessions", "models", "screenshots"]:
        if os.path.isdir(dir_name):
            print(f"     ✅ {dir_name}/")
        else:
            print(f"     ⚠️ {dir_name}/ — not created yet")

    # Summary
    print(f"\n  {'✅ All clear!' if issues == 0 else f'⚠️ {issues} issues found'}")


def pull_all_sessions():
    """Download all 6 session files from GitHub."""
    print("\n" + "=" * 60)
    print("📥 PULLING ALL SESSIONS FROM GITHUB")
    print("=" * 60)

    try:
        from sync import GitHubSync
        sync = GitHubSync()

        for name in SESSION_NAMES:
            remote_path = f"session_{name}.json"
            local_path = f"session_{name}.json"
            print(f"\n  Pulling {remote_path}...")
            sync._retry(lambda rp=remote_path, lp=local_path: sync._pull_file(rp, lp))

        # Also pull master model and review queue
        print(f"\n  Pulling master_model.json...")
        sync.pull_master()

        print(f"\n  Pulling review_queue.json...")
        sync.pull_review_queue()

        print("\n✅ All sessions pulled!")

    except ImportError:
        print("❌ sync.py not available")
    except Exception as e:
        print(f"❌ Error: {e}")


# ==============================================
# HELPER FUNCTIONS
# ==============================================

def _extract(data: dict, key1: str, key2: str = None, default=None):
    """Extract a value from nested session data."""
    if key1 in data:
        return data[key1]
    if key2 and key2 in data:
        return data[key2]
    for sub_key in ["session_stats", "metadata", "curriculum_state"]:
        sub = data.get(sub_key, {})
        if key1 in sub:
            return sub[key1]
        if key2 and key2 in sub:
            return sub[key2]
    return default


def _count_queue_items(session_name: str = None) -> int:
    """Count items in the review queue, optionally filtered by session."""
    if not os.path.exists("review_queue.json"):
        return 0
    try:
        with open("review_queue.json", 'r') as f:
            queue = json.load(f)
        if session_name:
            return sum(1 for m in queue if m.get("session") == session_name)
        return len(queue)
    except Exception:
        return 0


# ==============================================
# MAIN MENU
# ==============================================

def main():
    """Main menu loop."""
    print("\n" + "=" * 60)
    print("🎮 TERRITORIAL.IO AI — CONTROL CENTER")
    print("=" * 60)

    while True:
        print("\n  ┌─────────────────────────────────────┐")
        print("  │  1. 📋 Review moves                  │")
        print("  │  2. 🔄 Merge all sessions             │")
        print("  │  3. ⭐ Update best moves              │")
        print("  │  4. 📊 Show dashboard                 │")
        print("  │  5. 🏥 Health check                   │")
        print("  │  6. 📥 Pull all sessions              │")
        print("  │  7. 🚪 Exit                           │")
        print("  └─────────────────────────────────────┘")

        try:
            choice = input("\n  Enter choice (1-7): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if choice == "1":
            review_moves()
        elif choice == "2":
            merge_sessions()
        elif choice == "3":
            update_best_moves()
        elif choice == "4":
            show_dashboard()
        elif choice == "5":
            health_check()
        elif choice == "6":
            pull_all_sessions()
        elif choice == "7":
            print("\n👋 Goodbye!")
            break
        else:
            print("  ⚠️ Invalid choice — enter 1-7")


# ==============================================
# ENTRY POINT
# ==============================================

if __name__ == "__main__":
    main()
