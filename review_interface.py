"""
=============================================================================
review_interface.py — Terminal Human Move Review Interface
=============================================================================

A terminal-based interface for humans to rate AI moves. Loads moves
from review_queue.json, presents them one at a time, and lets the
human rate each of the 10 factors.

Rating options per factor:
    g = good
    b = can be better (NOT bad)
    x = bad
    s = skip (move stays in queue unchanged)

After rating, moves transfer from review_queue.json to rated_moves.json.
At the end of a review session, pushes rated_moves.json to GitHub.

Can be interrupted and resumed anytime without losing progress.

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

import json
import os
import sys
import time
from typing import List, Dict, Optional


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
    """Display a single move for human review."""
    print(f"\n{'='*60}")
    print(f"📋 MOVE REVIEW — Move {index + 1} of {total} | "
          f"Queue: {queue_remaining} remaining")
    print(f"{'='*60}")
    print()
    print(f"  🏷️  Move ID:     {move.get('move_id', 'unknown')}")
    print(f"  💻 Session:     {move.get('session', 'unknown')}")
    print(f"  🎮 Game:        {move.get('game_number', '?')}")
    print(f"  🎯 Move #:      {move.get('move_number', '?')}")
    print(f"  ⚡ Action:      {move.get('action_type', 'unknown')}")
    print(f"  📍 Action Index: {move.get('action_index', '?')}")

    grid_row = move.get('grid_row')
    grid_col = move.get('grid_col')
    if grid_row is not None and grid_col is not None:
        print(f"  📐 Grid:        row={grid_row}, col={grid_col}")

    print(f"\n  📊 Auto-Scored Factors:")
    auto_scores = move.get("auto_scores", {})

    # Handle nested format from MoveScorer (may have "factors" key)
    if isinstance(auto_scores, dict) and "factors" in auto_scores:
        auto_scores = auto_scores["factors"]

    for factor in FACTOR_NAMES:
        score = auto_scores.get(factor) if isinstance(auto_scores, dict) else None
        desc = FACTOR_DESCRIPTIONS.get(factor, "")
        if score is not None:
            bar = "█" * int(abs(score) * 10)
            sign = "+" if score >= 0 else "-"
            print(f"     {factor:30s} {sign}{abs(score):.2f} {bar}")
        else:
            print(f"     {factor:30s}  [NEEDS HUMAN RATING]")

    print()


def rate_move(move: Dict) -> Optional[Dict]:
    """
    Interactively rate a single move on all 10 factors.
    
    Returns:
        Dict of {factor_name: rating} if rated, or None if skipped
    """
    print("  Rate each factor: g=good, b=can be better, x=bad, s=skip move")
    print("  " + "-" * 55)

    ratings = {}

    for factor in FACTOR_NAMES:
        desc = FACTOR_DESCRIPTIONS.get(factor, "")

        while True:
            try:
                choice = input(f"  {factor:30s} ({desc[:40]:40s}) [g/b/x/s]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n\n⏸️  Review interrupted. Progress saved.")
                return None

            if choice == "s":
                # Skip entire move
                print("  ⏭️  Skipping this move (stays in queue)")
                return None

            if choice in RATING_MAP:
                ratings[factor] = RATING_MAP[choice]
                break
            else:
                print("     Invalid input — press g, b, x, or s")

    return ratings


def run_review_session():
    """
    Main review loop. Loads queue, presents moves one at a time,
    records ratings, and saves progress.
    """
    print("\n" + "=" * 60)
    print("📋 TERRITORIAL.IO — MOVE REVIEW INTERFACE")
    print("=" * 60)

    # Load the queue
    queue = load_queue()
    if not queue:
        print("\n✅ No moves to review! Queue is empty.")
        return

    rated_moves = load_rated()

    total_moves = len(queue)
    unique_sessions = len(set(m.get("session", "") for m in queue))
    print(f"\n  📊 Queue: {total_moves} moves from {unique_sessions} sessions")
    print(f"  📝 Already rated: {len(rated_moves)} moves")
    print()

    reviewed_count = 0
    skipped_count = 0
    moves_to_remove = []

    for i, move in enumerate(queue):
        queue_remaining = total_moves - i - skipped_count

        # Display the move
        display_move(move, i, total_moves, queue_remaining)

        # Get human rating
        ratings = rate_move(move)

        if ratings is None:
            # Move was skipped — stays in queue
            skipped_count += 1
            continue

        # Calculate overall rating
        good_count = sum(1 for r in ratings.values() if r == "good")
        bad_count = sum(1 for r in ratings.values() if r == "bad")
        if good_count >= 7:
            overall = "good"
        elif bad_count >= 4:
            overall = "bad"
        else:
            overall = "can_be_better"

        # Build rated entry
        rated_entry = {
            **move,
            "human_ratings": ratings,
            "overall_human_rating": overall,
            "rated_at": time.time(),
            "notes": None,
        }
        rated_moves.append(rated_entry)
        moves_to_remove.append(move["move_id"])
        reviewed_count += 1

        # Print feedback
        emoji = {"good": "✅", "can_be_better": "🔶", "bad": "❌"}
        print(f"\n  {emoji.get(overall, '?')} Rated: {overall} "
              f"({good_count}/10 good)")

        # Auto-save progress every 5 moves
        if reviewed_count % 5 == 0:
            _save_progress(queue, moves_to_remove, rated_moves)
            print(f"  💾 Progress auto-saved ({reviewed_count} rated so far)")

    # Final save
    _save_progress(queue, moves_to_remove, rated_moves)

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 REVIEW SESSION COMPLETE")
    print(f"{'='*60}")
    print(f"  ✅ Rated:   {reviewed_count} moves")
    print(f"  ⏭️  Skipped: {skipped_count} moves")
    print(f"  📋 Queue:   {total_moves - reviewed_count} remaining")

    # Push to GitHub if sync is available
    try:
        from sync import GitHubSync
        sync = GitHubSync()
        print("\n🔄 Syncing rated moves to GitHub...")
        sync.push_rated_moves()
        print("✅ Sync complete!")
    except ImportError:
        print("\n⚠️ sync.py not available — skipping GitHub push")
    except Exception as e:
        print(f"\n⚠️ Sync failed: {e}")


def _save_progress(queue: List[Dict], moves_to_remove: List[str],
                   rated_moves: List[Dict]):
    """Save current progress: update queue and rated files."""
    # Remove rated moves from queue
    updated_queue = [m for m in queue if m.get("move_id") not in moves_to_remove]
    save_queue(updated_queue)
    save_rated(rated_moves)


# ==============================================
# ENTRY POINT
# ==============================================

if __name__ == "__main__":
    run_review_session()
