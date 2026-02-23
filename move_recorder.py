"""
=============================================================================
move_recorder.py — Territorial.io Move Recording System
=============================================================================

Records every move made during training sessions. Each move gets a unique
ID and can be flagged for human review. Unrated moves are exported to
review_queue.json for the human rating interface.

Move ID format: "{session_name}_game{game_num}_move{move_num}"
Example: "colab1_game005_move047"

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

import json
import time
import os
from typing import List, Dict, Optional


# ==============================================
# CONFIGURATION
# ==============================================

REVIEW_QUEUE_FILE = "review_queue.json"
RATED_MOVES_FILE = "rated_moves.json"


# ==============================================
# MOVE RECORDER CLASS
# ==============================================

class MoveRecorder:
    """
    Records every AI move during training for later analysis and
    human rating.
    
    Each move gets a unique ID (e.g., "colab1_game005_move047") and
    can be flagged for human review via the review_queue.json.
    
    Usage:
        recorder = MoveRecorder("colab1")
        recorder.record_move(47, 5, action_dict, factor_scores)
        recorder.flag_for_review("colab1_game005_move047")
        recorder.export_unrated()  # Appends to review_queue.json
    """

    def __init__(self, session_name: str):
        """
        Initialize the move recorder for a training session.
        
        Args:
            session_name: Name of this session, e.g. "colab1", "kaggle2"
        """
        self.session_name = session_name
        self.moves: List[Dict] = []
        self.flagged_moves: List[str] = []
        self._move_index: Dict[str, Dict] = {}  # move_id → move dict

    def record_move(self, move_number: int, game_number: int,
                    action: dict, factor_scores: dict,
                    screenshot_path: Optional[str] = None) -> str:
        """
        Record a single move with all its metadata.
        
        Args:
            move_number:     Move number within the game (1-indexed)
            game_number:     Game number within the session (1-indexed)
            action:          Action dictionary from GameAgent
            factor_scores:   Dictionary of 10 factor scores from MoveScorer
            screenshot_path: Optional path to saved screenshot
            
        Returns:
            The unique move ID string
        """
        # Generate unique move ID
        move_id = f"{self.session_name}_game{game_number:03d}_move{move_number:03d}"

        move_record = {
            "move_id": move_id,
            "session": self.session_name,
            "game_number": game_number,
            "move_number": move_number,
            "action_type": action.get("action_type", "unknown"),
            "action_index": action.get("action_index", 0),
            "grid_row": action.get("grid_row"),
            "grid_col": action.get("grid_col"),
            "screen_x": action.get("screen_x"),
            "screen_y": action.get("screen_y"),
            "was_random": action.get("was_random", False),
            "confidence": action.get("confidence", 0.0),
            "auto_scores": factor_scores,
            "human_ratings": None,  # Not yet rated
            "screenshot_path": screenshot_path,
            "timestamp": time.time(),
            "priority": 0.5,  # Default priority (0.0 to 1.0)
            "flagged_for_review": False,
        }

        self.moves.append(move_record)
        self._move_index[move_id] = move_record

        return move_id

    def flag_for_review(self, move_id: str) -> bool:
        """
        Mark a move as needing human rating.
        
        Args:
            move_id: The unique move ID to flag
            
        Returns:
            True if successfully flagged, False if move_id not found
        """
        if move_id in self._move_index:
            self._move_index[move_id]["flagged_for_review"] = True
            if move_id not in self.flagged_moves:
                self.flagged_moves.append(move_id)
            return True
        return False

    def export_unrated(self, filepath: str = REVIEW_QUEUE_FILE):
        """
        Append all unrated moves to review_queue.json.
        
        IMPORTANT: This APPENDS to the existing file rather than
        overwriting, so moves from other sessions are preserved.
        
        Args:
            filepath: Path to the review queue JSON file
        """
        # Load existing queue if it exists
        existing_queue = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing_queue = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_queue = []

        # Collect existing move IDs to avoid duplicates
        existing_ids = {entry["move_id"] for entry in existing_queue}

        # Add our unrated moves
        new_entries = 0
        for move in self.moves:
            if move["human_ratings"] is None and move["move_id"] not in existing_ids:
                queue_entry = {
                    "move_id": move["move_id"],
                    "session": move["session"],
                    "game_number": move["game_number"],
                    "move_number": move["move_number"],
                    "action_type": move["action_type"],
                    "action_index": move["action_index"],
                    "grid_row": move["grid_row"],
                    "grid_col": move["grid_col"],
                    "auto_scores": move["auto_scores"],
                    "human_ratings": None,
                    "timestamp": move["timestamp"],
                    "priority": move["priority"],
                }
                existing_queue.append(queue_entry)
                new_entries += 1

        # Save updated queue
        with open(filepath, 'w') as f:
            json.dump(existing_queue, f, indent=2)

        print(f"   📋 Exported {new_entries} unrated moves to {filepath} "
              f"(total queue: {len(existing_queue)})")

    def mark_rated(self, move_id: str, ratings_dict: dict,
                   filepath: str = RATED_MOVES_FILE):
        """
        Mark a move as rated by a human. Moves the entry from the
        review queue to rated_moves.json.
        
        Args:
            move_id:      The unique move ID
            ratings_dict: Dict of {factor_name: "good"/"can_be_better"/"bad"}
            filepath:     Path to the rated moves JSON file
        """
        # Load existing rated moves
        rated_moves = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    rated_moves = json.load(f)
            except (json.JSONDecodeError, IOError):
                rated_moves = []

        # Find the move in our records
        if move_id in self._move_index:
            move = self._move_index[move_id].copy()
        else:
            move = {"move_id": move_id}

        # Count good ratings
        good_count = sum(1 for r in ratings_dict.values() if r == "good")
        bad_count = sum(1 for r in ratings_dict.values() if r == "bad")

        if good_count >= 7:
            overall = "good"
        elif bad_count >= 4:
            overall = "bad"
        else:
            overall = "can_be_better"

        # Build rated entry
        rated_entry = {
            **move,
            "human_ratings": ratings_dict,
            "overall_human_rating": overall,
            "rated_at": time.time(),
            "notes": None,
        }

        rated_moves.append(rated_entry)

        # Save
        with open(filepath, 'w') as f:
            json.dump(rated_moves, f, indent=2)

        # Update internal record
        if move_id in self._move_index:
            self._move_index[move_id]["human_ratings"] = ratings_dict

    def save_session(self, filepath: str):
        """
        Save all recorded moves for this session to a JSON file.
        
        Args:
            filepath: Path to save the session moves file
        """
        session_data = {
            "session_name": self.session_name,
            "total_moves": len(self.moves),
            "flagged_for_review": len(self.flagged_moves),
            "unrated_count": sum(1 for m in self.moves if m["human_ratings"] is None),
            "saved_at": time.time(),
            "moves": self.moves,
        }

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        print(f"   💾 Session moves saved: {filepath} ({len(self.moves)} moves)")

    def get_move(self, move_id: str) -> Optional[Dict]:
        """Get a move by its ID."""
        return self._move_index.get(move_id)

    def get_unrated_count(self) -> int:
        """Count how many moves haven't been rated yet."""
        return sum(1 for m in self.moves if m["human_ratings"] is None)


# ==============================================
# DEMO
# ==============================================

def demo():
    """Quick demo of the move recorder."""
    print("\n📝 Move Recorder Demo")
    print("=" * 50)

    recorder = MoveRecorder("colab1")

    # Record some test moves
    for game in range(1, 3):
        for move in range(1, 6):
            action = {
                "action_type": "click",
                "action_index": game * 100 + move,
                "grid_row": move,
                "grid_col": game + move,
                "screen_x": 0.5,
                "screen_y": 0.5,
                "was_random": move % 2 == 0,
                "confidence": 0.8 if move % 2 != 0 else 0.0,
            }
            scores = {
                "territory_change": 0.3,
                "attack_timing": None,
                "target_selection": 0.5,
                "border_efficiency": None,
                "survival_instinct": 0.7,
                "power_management": None,
                "opportunity_recognition": 0.2,
                "multi_enemy_awareness": None,
                "map_position": 0.4,
                "aggression_balance": None,
            }
            move_id = recorder.record_move(move, game, action, scores)
            print(f"   Recorded: {move_id}")

    # Flag a move for review
    recorder.flag_for_review("colab1_game001_move003")

    print(f"\n   Total moves: {len(recorder.moves)}")
    print(f"   Unrated: {recorder.get_unrated_count()}")
    print(f"   Flagged: {len(recorder.flagged_moves)}")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
