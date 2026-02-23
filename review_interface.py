"""
=============================================================================
review_interface.py — Jupyter/Terminal Move Review Interface
=============================================================================

Updated to fix "Blind Rating" in Kaggle/Colab.
Now displays the game screenshot inline using IPython.display if available.

Usage:
    python review_interface.py
"""

# ==============================================
# IMPORTS
# ==============================================

import json
import os
import sys
import time
from typing import List, Dict, Optional

# Check if running in a notebook environment
def is_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

# Import display libraries if in notebook
if is_notebook():
    from IPython.display import display, Image as IPImage, clear_output

# ==============================================
# CONFIGURATION
# ==============================================

REVIEW_QUEUE_FILE = "review_queue.json"
RATED_MOVES_FILE = "rated_moves.json"

FACTOR_NAMES = [
    "territory_change",
    "attack_timing",
    "target_selection",
    "border_efficiency",
    "survival_instinct",
    "power_management",
    "opportunity_recognition",
    "multi_enemy_awareness",
    "map_position",
    "aggression_balance",
]

FACTOR_DESCRIPTIONS = {
    "territory_change":        "Did we gain or lose territory?",
    "attack_timing":           "Was this the right moment to attack?",
    "target_selection":        "Did we pick the right target?",
    "border_efficiency":       "Are our borders clean and compact?",
    "survival_instinct":       "Did we avoid danger and stay alive?",
    "power_management":        "Did we use troop power wisely?",
    "opportunity_recognition": "Did we exploit openings?",
    "multi_enemy_awareness":   "Were we aware of all threats?",
    "map_position":            "Is our territory in a strong position?",
    "aggression_balance":      "Right level of aggression?",
}

RATING_MAP = {
    "g": "good",
    "b": "can_be_better",
    "x": "bad",
}


# ==============================================
# REVIEW INTERFACE
# ==============================================

def load_queue(filepath: str = REVIEW_QUEUE_FILE) -> List[Dict]:
    """Load the review queue from JSON file."""
    if not os.path.exists(filepath):
        print(f"⚠️ Review queue file not found: {filepath}")
        return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ Error loading review queue: {e}")
        return []


def save_queue(queue: List[Dict], filepath: str = REVIEW_QUEUE_FILE):
    """Save the review queue to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(queue, f, indent=2)


def load_rated(filepath: str = RATED_MOVES_FILE) -> List[Dict]:
    """Load already-rated moves."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_rated(rated: List[Dict], filepath: str = RATED_MOVES_FILE):
    """Save rated moves to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(rated, f, indent=2)


def display_move(move: Dict, index: int, total: int, queue_remaining: int):
    """Display a single move with Screenshot support for Notebooks."""
    
    # Clear previous cell output if in notebook to keep it clean
    if is_notebook():
        clear_output(wait=True)

    print(f"\n{'='*60}")
    print(f"📋 MOVE REVIEW — Move {index + 1} of {total} | "
          f"Queue: {queue_remaining} remaining")
    print(f"{'='*60}")
    print()
    print(f"  🏷️  Move ID:     {move.get('move_id', 'unknown')}")
    print(f"  🎮 Game:        {move.get('game_number', '?')}")
    print(f"  ⚡ Action:      {move.get('action_type', 'unknown')}")

    grid_row = move.get('grid_row')
    grid_col = move.get('grid_col')
    if grid_row is not None and grid_col is not None:
        print(f"  📐 Grid:        row={grid_row}, col={grid_col}")

    # --- IMAGE DISPLAY LOGIC ---
    screenshot_path = move.get("screenshot_path")
    
    if screenshot_path and os.path.exists(screenshot_path):
        if is_notebook():
            print(f"\n  🖼️ SCREENSHOT ({screenshot_path}):")
            try:
                display(IPImage(filename=screenshot_path))
            except Exception as e:
                print(f"  ⚠️ Could not render image inline: {e}")
        else:
            print(f"\n  🖼️ Screenshot available at: {screenshot_path}")
            print(f"     (Open file manually if not using a Notebook)")
    elif screenshot_path:
        print(f"\n  ⚠️ Screenshot file missing: {screenshot_path}")
    else:
        print("\n  ⚠️ No screenshot captured for this move.")
    
    # ---------------------------

    print(f"\n  📊 Auto-Scored Factors:")
    auto_scores = move.get("auto_scores", {})
    if isinstance(auto_scores, dict) and "factors" in auto_scores:
        auto_scores = auto_scores["factors"]

    for factor in FACTOR_NAMES:
        score = auto_scores.get(factor) if isinstance(auto_scores, dict) else None
        if score is not None:
            bar = "█" * int(abs(score) * 10)
            sign = "+" if score >= 0 else "-"
            print(f"     {factor:30s} {sign}{abs(score):.2f} {bar}")
        else:
            print(f"     {factor:30s}  [NEEDS HUMAN RATING]")
    print()


def rate_move(move: Dict) -> Optional[Dict]:
    """Interactively rate a single move."""
    print("  Rate each factor: g=good, b=can be better, x=bad, s=skip move")
    print("  " + "-" * 55)

    ratings = {}

    for factor in FACTOR_NAMES:
        desc = FACTOR_DESCRIPTIONS.get(factor, "")

        while True:
            try:
                prompt_text = f"  {factor:30s} ({desc[:40]:40s}) [g/b/x/s]: "
                choice = input(prompt_text).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n\n⏸️  Review interrupted. Progress saved.")
                return None

            if choice == "s":
                print("  ⏭️  Skipping this move (stays in queue)")
                return None

            if choice in RATING_MAP:
                ratings[factor] = RATING_MAP[choice]
                break
            else:
                print("     Invalid input — press g, b, x, or s")

    return ratings


def run_review_session():
    print("\n" + "=" * 60)
    print("📋 TERRITORIAL.IO — MOVE REVIEW INTERFACE")
    print(f"   Mode: {'Jupyter Notebook' if is_notebook() else 'Terminal'}")
    print("=" * 60)

    queue = load_queue()
    if not queue:
        print("\n✅ No moves to review! Queue is empty.")
        return

    rated_moves = load_rated()
    total_moves = len(queue)
    
    reviewed_count = 0
    skipped_count = 0
    moves_to_remove = []

    for i, move in enumerate(queue):
        # Stop if we processed 50 moves to prevent session timeout/fatigue
        if reviewed_count >= 50:
            print("\n🛑 Pausing after 50 moves. Run again to continue!")
            break

        queue_remaining = total_moves - i - skipped_count

        display_move(move, i, total_moves, queue_remaining)
        ratings = rate_move(move)

        if ratings is None:
            skipped_count += 1
            continue

        # Simple Logic for overall score
        good_count = sum(1 for r in ratings.values() if r == "good")
        bad_count = sum(1 for r in ratings.values() if r == "bad")
        
        if good_count >= 7: overall = "good"
        elif bad_count >= 4: overall = "bad"
        else: overall = "can_be_better"

        rated_entry = {
            **move,
            "human_ratings": ratings,
            "overall_human_rating": overall,
            "rated_at": time.time(),
        }
        
        rated_moves.append(rated_entry)
        moves_to_remove.append(move["move_id"])
        reviewed_count += 1

        print(f"\n  ✅ Rated: {overall}")

        if reviewed_count % 5 == 0:
            _save_progress(queue, moves_to_remove, rated_moves)
            print(f"  💾 Progress auto-saved")

    _save_progress(queue, moves_to_remove, rated_moves)

    # Push to GitHub
    try:
        from sync import GitHubSync
        sync = GitHubSync()
        print("\n🔄 Syncing rated moves to GitHub...")
        sync.push_rated_moves()
        print("✅ Sync complete!")
    except ImportError:
        pass
    except Exception as e:
        print(f"Sync error: {e}")

def _save_progress(queue, remove_ids, rated):
    updated_queue = [m for m in queue if m.get("move_id") not in remove_ids]
    save_queue(updated_queue)
    save_rated(rated)

if __name__ == "__main__":
    run_review_session()