"""
================================================================================
📚 CURRICULUM.PY — TRAINING PHASE MANAGER FOR TERRITORIAL.IO AI
================================================================================

This file manages HOW the AI learns over time:
  - Phase 1 (Games 1-1000):   Human rates every move, missed ratings queued
  - Phase 2 (Games 1000+):    Win/loss rewards take over as main teacher
  - Difficulty auto-adjusts:  Easy → Medium → Hard based on win rate
  - Tracks readiness to advance based on performance metrics

Think of this as the AI's "school system" — it starts in kindergarten
and graduates when it proves it's ready.

Used by: trainer.py (checks phase & difficulty before each game)
================================================================================
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ==============================================================================
# 🎯 TRAINING PHASES — The AI's learning journey
# ==============================================================================

class TrainingPhase(Enum):
    """
    The two main phases of training:
    
    PHASE_1 (Human Teacher):
        - Games 1 through 1000
        - Every single move gets a human rating
        - If human misses rating a move, it goes to a review queue
        - AI learns from human expertise
        - Like having a tutor sit next to you
    
    PHASE_2 (Self Learning):
        - Games 1000+
        - Win = big reward, Loss = big penalty
        - AI teaches itself through trial and error
        - Human can still rate occasionally but it's optional
        - Like graduating and learning from experience
    """
    PHASE_1_HUMAN_TEACHER = "phase_1_human_teacher"
    PHASE_2_SELF_LEARNING = "phase_2_self_learning"


# ==============================================================================
# 🎮 DIFFICULTY LEVELS — How hard the opponents are
# ==============================================================================

class Difficulty(Enum):
    """
    Three difficulty levels that auto-progress:
    
    EASY:
        - Few AI opponents (2-3 bots)
        - Bots are passive, rarely attack
        - Small maps so games are short
        - Good for learning basic territory control
    
    MEDIUM:
        - More opponents (4-6 bots)
        - Bots are moderately aggressive
        - Medium maps with more strategy needed
        - Good for learning attack timing
    
    HARD:
        - Many opponents (7-10 bots)
        - Bots are very aggressive and smart
        - Large maps with complex strategy
        - Good for learning advanced tactics
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ==============================================================================
# 📊 PROMOTION THRESHOLDS — When the AI levels up
# ==============================================================================

# Win rates needed to advance to next difficulty
# (measured over the last N games at current difficulty)
PROMOTION_THRESHOLDS = {
    # Need 60% win rate on Easy to move to Medium
    Difficulty.EASY: {
        "win_rate_needed": 0.60,        # 60% wins required
        "min_games_played": 50,          # Must play at least 50 games
        "min_consecutive_wins": 5,       # Must win 5 in a row at least once
        "description": "Win 60% of 50+ Easy games to unlock Medium"
    },
    
    # Need 55% win rate on Medium to move to Hard
    Difficulty.MEDIUM: {
        "win_rate_needed": 0.55,        # 55% wins required (medium is harder)
        "min_games_played": 100,         # Must play at least 100 games
        "min_consecutive_wins": 4,       # Must win 4 in a row
        "description": "Win 55% of 100+ Medium games to unlock Hard"
    },
    
    # Hard is the final level — no promotion, just mastery
    Difficulty.HARD: {
        "win_rate_needed": 1.0,         # Can't promote past Hard
        "min_games_played": 999999,      # Never promotes
        "min_consecutive_wins": 999,     # Never promotes
        "description": "Hard is the final difficulty — master it!"
    }
}

# When to move from Phase 1 (human teacher) to Phase 2 (self learning)
PHASE_TRANSITION_THRESHOLD = {
    "total_games_needed": 1000,          # After 1000 games with human ratings
    "min_win_rate": 0.30,                # Must win at least 30% of games
    "min_rated_moves": 5000,             # Must have 5000+ human-rated moves
    "description": "Play 1000 games with human ratings to enter Phase 2"
}


# ==============================================================================
# 📋 REVIEW QUEUE — Moves the human missed rating
# ==============================================================================

@dataclass
class UnratedMove:
    """
    When a human misses rating a move during Phase 1,
    it goes into this review queue so they can rate it later.
    
    Each unrated move stores:
      - game_id:        Which game it happened in
      - move_number:    Which move in the game (1st, 2nd, etc.)
      - screenshot:     What the game looked like (as file path)
      - action_taken:   What the AI did
      - timestamp:      When it happened
      - priority:       How important this move is to rate
                        (critical moments get higher priority)
    """
    game_id: int                         # Which game
    move_number: int                     # Which move in that game
    screenshot_path: str                 # Path to saved screenshot
    action_taken: int                    # Action index the AI chose
    timestamp: float                     # Unix timestamp
    priority: float = 0.5               # 0.0 = low priority, 1.0 = urgent
    context: str = ""                    # Optional note about the situation
    
    def to_dict(self) -> dict:
        """Convert to dictionary for saving to JSON."""
        return asdict(self)


# ==============================================================================
# 📊 PERFORMANCE TRACKER — Stats for current difficulty
# ==============================================================================

@dataclass
class DifficultyStats:
    """
    Tracks performance at ONE difficulty level.
    Used to decide when to promote to next difficulty.
    
    Example:
      On Easy difficulty, tracks:
        - Total games played on Easy
        - Wins and losses on Easy
        - Current win streak on Easy
        - Best win streak ever on Easy
    """
    difficulty: str                              # "easy", "medium", or "hard"
    games_played: int = 0                        # Total games at this level
    wins: int = 0                                # Total wins
    losses: int = 0                              # Total losses
    current_win_streak: int = 0                  # Current consecutive wins
    best_win_streak: int = 0                     # Best ever consecutive wins
    current_loss_streak: int = 0                 # Current consecutive losses
    worst_loss_streak: int = 0                   # Worst ever consecutive losses
    total_moves_made: int = 0                    # Total moves across all games
    total_territory_gained: float = 0.0          # Sum of all territory gains
    avg_game_length: float = 0.0                 # Average moves per game
    last_10_results: List[bool] = field(         # Last 10 game results
        default_factory=list                     # True = win, False = loss
    )
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate as a percentage (0.0 to 1.0)."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    @property
    def recent_win_rate(self) -> float:
        """Win rate over last 10 games (more recent = more relevant)."""
        if len(self.last_10_results) == 0:
            return 0.0
        return sum(self.last_10_results) / len(self.last_10_results)
    
    def record_game(self, won: bool, moves_made: int, territory_gained: float):
        """
        Record the result of one game at this difficulty.
        
        Args:
            won:               Did the AI win this game?
            moves_made:        How many moves in the game
            territory_gained:  Net territory change
        """
        # --- Update basic counters ---
        self.games_played += 1
        self.total_moves_made += moves_made
        self.total_territory_gained += territory_gained
        
        if won:
            self.wins += 1
            self.current_win_streak += 1
            self.current_loss_streak = 0
            if self.current_win_streak > self.best_win_streak:
                self.best_win_streak = self.current_win_streak
        else:
            self.losses += 1
            self.current_loss_streak += 1
            self.current_win_streak = 0
            if self.current_loss_streak > self.worst_loss_streak:
                self.worst_loss_streak = self.current_loss_streak
        
        # --- Update last 10 results (sliding window) ---
        self.last_10_results.append(won)
        if len(self.last_10_results) > 10:
            self.last_10_results.pop(0)
        
        # --- Update average game length ---
        self.avg_game_length += (moves_made - self.avg_game_length) / self.games_played
    
    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return asdict(self)


# ==============================================================================
# 🎓 CURRICULUM MANAGER — The main controller
# ==============================================================================

class CurriculumManager:
    """
    The main class that manages the AI's entire learning journey.
    
    It controls:
      1. Which PHASE the AI is in (human teacher vs self learning)
      2. Which DIFFICULTY the AI plays at (easy/medium/hard)
      3. The REVIEW QUEUE of unrated moves
      4. PROMOTION decisions (when to level up)
    
    Usage:
        curriculum = CurriculumManager()
        
        # Check current state
        phase = curriculum.get_current_phase()
        difficulty = curriculum.get_current_difficulty()
        
        # After a game
        curriculum.record_game_result(won=True, moves=50, territory=0.3)
        
        # Check if ready to advance
        if curriculum.should_promote():
            curriculum.promote_difficulty()
    """
    
    def __init__(self, save_path: str = "curriculum_state.json"):
        """
        Initialize the curriculum manager.
        
        Args:
            save_path: Where to save/load curriculum state
        """
        self.save_path = save_path
        
        # --- Current state ---
        self.current_phase = TrainingPhase.PHASE_1_HUMAN_TEACHER
        self.current_difficulty = Difficulty.EASY
        self.total_games_played = 0
        self.total_human_rated_moves = 0
        
        # --- Performance tracking per difficulty ---
        self.difficulty_stats: Dict[str, DifficultyStats] = {
            Difficulty.EASY.value: DifficultyStats(difficulty=Difficulty.EASY.value),
            Difficulty.MEDIUM.value: DifficultyStats(difficulty=Difficulty.MEDIUM.value),
            Difficulty.HARD.value: DifficultyStats(difficulty=Difficulty.HARD.value),
        }
        
        # --- Review queue for missed ratings (Phase 1) ---
        self.review_queue: List[UnratedMove] = []
        self.max_queue_size = 500
        
        # --- History of phase/difficulty changes ---
        self.transition_history: List[Dict] = []
        
        # --- Try to load saved state ---
        self._load_state()
    
    # ==========================================================================
    # 📍 GETTERS — Check current state
    # ==========================================================================
    
    def get_current_phase(self) -> TrainingPhase:
        """Get the current training phase."""
        return self.current_phase
    
    def get_current_difficulty(self) -> Difficulty:
        """Get the current difficulty level."""
        return self.current_difficulty
    
    def get_current_stats(self) -> DifficultyStats:
        """Get performance stats for the current difficulty."""
        return self.difficulty_stats[self.current_difficulty.value]
    
    def get_phase_description(self) -> str:
        """Get a human-readable description of current state."""
        phase_name = (
            "Phase 1: Human Teacher" 
            if self.current_phase == TrainingPhase.PHASE_1_HUMAN_TEACHER
            else "Phase 2: Self Learning"
        )
        stats = self.get_current_stats()
        
        return (
            f"\n{'='*60}\n"
            f"📚 CURRICULUM STATUS\n"
            f"{'='*60}\n"
            f"Phase:           {phase_name}\n"
            f"Difficulty:      {self.current_difficulty.value.upper()}\n"
            f"Total Games:     {self.total_games_played}\n"
            f"Games at {self.current_difficulty.value}: {stats.games_played}\n"
            f"Win Rate:        {stats.win_rate:.1%}\n"
            f"Recent Win Rate: {stats.recent_win_rate:.1%}\n"
            f"Best Streak:     {stats.best_win_streak} wins\n"
            f"Review Queue:    {len(self.review_queue)} unrated moves\n"
            f"Human Ratings:   {self.total_human_rated_moves}\n"
            f"{'='*60}"
        )
    
    # ==========================================================================
    # 🎮 GAME RECORDING — Track results
    # ==========================================================================
    
    def record_game_result(
        self,
        won: bool,
        moves_made: int,
        territory_gained: float,
        unrated_moves: Optional[List[UnratedMove]] = None
    ):
        """
        Record the result of a completed game.
        
        This is called after EVERY game. It:
          1. Updates stats for the current difficulty
          2. Adds any unrated moves to the review queue
          3. Checks if AI should be promoted
          4. Checks if AI should switch phases
        
        Args:
            won:              Did the AI win?
            moves_made:       Total moves in the game
            territory_gained: Net territory change (0.0 to 1.0)
            unrated_moves:    List of moves the human didn't rate
        """
        # --- Step 1: Update total games ---
        self.total_games_played += 1
        
        # --- Step 2: Update difficulty-specific stats ---
        stats = self.get_current_stats()
        stats.record_game(won, moves_made, territory_gained)
        
        # --- Step 3: Add unrated moves to review queue ---
        if unrated_moves:
            for move in unrated_moves:
                self._add_to_review_queue(move)
        
        # --- Step 4: Check for promotion ---
        if self.should_promote():
            self.promote_difficulty()
        
        # --- Step 5: Check for phase transition ---
        if self.should_transition_phase():
            self.transition_to_phase_2()
        
        # --- Step 6: Auto-save ---
        self._save_state()
        
        # --- Step 7: Print update ---
        result_emoji = "🏆" if won else "💀"
        print(
            f"{result_emoji} Game {self.total_games_played} complete | "
            f"Difficulty: {self.current_difficulty.value} | "
            f"Win Rate: {stats.win_rate:.1%} | "
            f"Streak: {stats.current_win_streak}W / {stats.current_loss_streak}L"
        )
    
    def record_human_rating(self, count: int = 1):
        """Record that the human rated some moves."""
        self.total_human_rated_moves += count
    
    # ==========================================================================
    # 📈 PROMOTION LOGIC — Difficulty advancement
    # ==========================================================================
    
    def should_promote(self) -> bool:
        """
        Check if the AI is ready to move to a harder difficulty.
        
        Checks three things:
          1. Has played enough games at current difficulty
          2. Win rate is above the threshold
          3. Has achieved the required consecutive win streak
        
        Returns:
            True if AI should be promoted
        """
        # Can't promote past Hard
        if self.current_difficulty == Difficulty.HARD:
            return False
        
        threshold = PROMOTION_THRESHOLDS[self.current_difficulty]
        stats = self.get_current_stats()
        
        # --- Check 1: Enough games played ---
        if stats.games_played < threshold["min_games_played"]:
            return False
        
        # --- Check 2: Win rate high enough ---
        if stats.win_rate < threshold["win_rate_needed"]:
            return False
        
        # --- Check 3: Had a good enough win streak ---
        if stats.best_win_streak < threshold["min_consecutive_wins"]:
            return False
        
        return True
    
    def promote_difficulty(self):
        """
        Move the AI to the next difficulty level.
        EASY → MEDIUM → HARD
        """
        old_difficulty = self.current_difficulty
        
        if self.current_difficulty == Difficulty.EASY:
            self.current_difficulty = Difficulty.MEDIUM
        elif self.current_difficulty == Difficulty.MEDIUM:
            self.current_difficulty = Difficulty.HARD
        else:
            return
        
        transition = {
            "type": "difficulty_promotion",
            "from": old_difficulty.value,
            "to": self.current_difficulty.value,
            "at_game": self.total_games_played,
            "timestamp": time.time(),
            "win_rate_at_promotion": self.difficulty_stats[old_difficulty.value].win_rate
        }
        self.transition_history.append(transition)
        
        print(f"\n{'🎉'*20}")
        print(f"🎓 PROMOTED! {old_difficulty.value.upper()} → {self.current_difficulty.value.upper()}")
        print(f"   After {self.total_games_played} total games")
        print(f"   Win rate was: {self.difficulty_stats[old_difficulty.value].win_rate:.1%}")
        print(f"{'🎉'*20}\n")
        
        self._save_state()
    
    def get_promotion_progress(self) -> Dict:
        """
        Get a progress report toward next promotion.
        Returns dictionary with progress percentages for each requirement.
        """
        if self.current_difficulty == Difficulty.HARD:
            return {"status": "MAX_DIFFICULTY", "message": "Already at Hard!"}
        
        threshold = PROMOTION_THRESHOLDS[self.current_difficulty]
        stats = self.get_current_stats()
        
        return {
            "current_difficulty": self.current_difficulty.value,
            "next_difficulty": (
                Difficulty.MEDIUM.value if self.current_difficulty == Difficulty.EASY 
                else Difficulty.HARD.value
            ),
            "games_played": {
                "current": stats.games_played,
                "needed": threshold["min_games_played"],
                "progress": min(1.0, stats.games_played / threshold["min_games_played"])
            },
            "win_rate": {
                "current": stats.win_rate,
                "needed": threshold["win_rate_needed"],
                "progress": min(1.0, stats.win_rate / threshold["win_rate_needed"])
            },
            "win_streak": {
                "current": stats.best_win_streak,
                "needed": threshold["min_consecutive_wins"],
                "progress": min(1.0, stats.best_win_streak / threshold["min_consecutive_wins"])
            },
            "overall_progress": min(1.0, (
                min(1.0, stats.games_played / threshold["min_games_played"]) * 0.3 +
                min(1.0, stats.win_rate / threshold["win_rate_needed"]) * 0.5 +
                min(1.0, stats.best_win_streak / threshold["min_consecutive_wins"]) * 0.2
            ))
        }
    
    # ==========================================================================
    # 🔄 PHASE TRANSITION — Human teacher → Self learning
    # ==========================================================================
    
    def should_transition_phase(self) -> bool:
        """
        Check if the AI is ready to move from Phase 1 to Phase 2.
        
        Requirements:
          1. Played at least 1000 total games
          2. Win rate is at least 30%
          3. Has at least 5000 human-rated moves
        """
        if self.current_phase == TrainingPhase.PHASE_2_SELF_LEARNING:
            return False
        
        threshold = PHASE_TRANSITION_THRESHOLD
        
        total_wins = sum(
            stats.wins for stats in self.difficulty_stats.values()
        )
        overall_win_rate = (
            total_wins / self.total_games_played 
            if self.total_games_played > 0 
            else 0.0
        )
        
        games_ok = self.total_games_played >= threshold["total_games_needed"]
        winrate_ok = overall_win_rate >= threshold["min_win_rate"]
        ratings_ok = self.total_human_rated_moves >= threshold["min_rated_moves"]
        
        return games_ok and winrate_ok and ratings_ok
    
    def transition_to_phase_2(self):
        """
        Move from Phase 1 (human teacher) to Phase 2 (self learning).
        This is a big moment! The AI is now learning on its own.
        """
        if self.current_phase == TrainingPhase.PHASE_2_SELF_LEARNING:
            return
        
        self.current_phase = TrainingPhase.PHASE_2_SELF_LEARNING
        
        transition = {
            "type": "phase_transition",
            "from": "phase_1",
            "to": "phase_2",
            "at_game": self.total_games_played,
            "timestamp": time.time(),
            "human_rated_moves": self.total_human_rated_moves
        }
        self.transition_history.append(transition)
        
        print(f"\n{'🎓'*20}")
        print(f"🎓 PHASE TRANSITION! Human Teacher → Self Learning")
        print(f"   After {self.total_games_played} games")
        print(f"   With {self.total_human_rated_moves} human-rated moves")
        print(f"   The AI is now learning on its own!")
        print(f"{'🎓'*20}\n")
        
        self._save_state()
    
    # ==========================================================================
    # 📋 REVIEW QUEUE — Unrated moves from Phase 1
    # ==========================================================================
    
    def _add_to_review_queue(self, move: UnratedMove):
        """Add an unrated move to the review queue."""
        if len(self.review_queue) >= self.max_queue_size:
            self.review_queue.sort(key=lambda m: m.priority)
            self.review_queue.pop(0)
        self.review_queue.append(move)
    
    def get_moves_to_review(self, count: int = 10) -> List[UnratedMove]:
        """Get highest priority unrated moves for human review."""
        sorted_queue = sorted(
            self.review_queue, 
            key=lambda m: m.priority, 
            reverse=True
        )
        return sorted_queue[:count]
    
    def mark_as_reviewed(self, game_id: int, move_number: int):
        """Remove a move from the review queue after human rates it."""
        self.review_queue = [
            move for move in self.review_queue
            if not (move.game_id == game_id and move.move_number == move_number)
        ]
        self.total_human_rated_moves += 1
    
    def get_queue_size(self) -> int:
        """Get how many moves are waiting for review."""
        return len(self.review_queue)
    
    # ==========================================================================
    # 🎯 TRAINING CONFIG — What settings to use right now
    # ==========================================================================
    
    def get_training_config(self) -> Dict:
        """
        Get the current training configuration based on phase and difficulty.
        Returns settings that trainer.py should use for the next game.
        """
        difficulty_configs = {
            Difficulty.EASY: {
                "num_opponents": 3,
                "map_size": "small",
                "learning_rate": 0.001,
                "exploration_rate": 0.3,
                "batch_size": 32,
                "discount_factor": 0.9,
            },
            Difficulty.MEDIUM: {
                "num_opponents": 5,
                "map_size": "medium",
                "learning_rate": 0.0005,
                "exploration_rate": 0.15,
                "batch_size": 64,
                "discount_factor": 0.95,
            },
            Difficulty.HARD: {
                "num_opponents": 8,
                "map_size": "large",
                "learning_rate": 0.0001,
                "exploration_rate": 0.05,
                "batch_size": 128,
                "discount_factor": 0.99,
            }
        }
        
        config = difficulty_configs[self.current_difficulty].copy()
        
        if self.current_phase == TrainingPhase.PHASE_1_HUMAN_TEACHER:
            config["reward_source"] = "human_ratings"
            config["require_human_rating"] = True
            config["human_rating_weight"] = 0.8
            config["auto_reward_weight"] = 0.2
        else:
            config["reward_source"] = "auto_rewards"
            config["require_human_rating"] = False
            config["human_rating_weight"] = 0.2
            config["auto_reward_weight"] = 0.8
        
        config["phase"] = self.current_phase.value
        config["difficulty"] = self.current_difficulty.value
        config["total_games"] = self.total_games_played
        
        return config
    
    # ==========================================================================
    # 💾 SAVE & LOAD — Persist curriculum state
    # ==========================================================================
    
    def _save_state(self):
        """Save the entire curriculum state to a JSON file."""
        state = {
            "current_phase": self.current_phase.value,
            "current_difficulty": self.current_difficulty.value,
            "total_games_played": self.total_games_played,
            "total_human_rated_moves": self.total_human_rated_moves,
            "difficulty_stats": {
                k: v.to_dict() for k, v in self.difficulty_stats.items()
            },
            "review_queue": [m.to_dict() for m in self.review_queue],
            "transition_history": self.transition_history,
            "saved_at": time.time()
        }
        
        try:
            with open(self.save_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save curriculum state: {e}")
    
    def _load_state(self):
        """Load curriculum state from JSON file if it exists."""
        if not os.path.exists(self.save_path):
            print("📚 No saved curriculum found — starting fresh!")
            return
        
        try:
            with open(self.save_path, 'r') as f:
                state = json.load(f)
            
            self.current_phase = TrainingPhase(state["current_phase"])
            self.current_difficulty = Difficulty(state["current_difficulty"])
            self.total_games_played = state["total_games_played"]
            self.total_human_rated_moves = state.get("total_human_rated_moves", 0)
            
            for key, stats_dict in state.get("difficulty_stats", {}).items():
                if key in self.difficulty_stats:
                    stats = self.difficulty_stats[key]
                    stats.games_played = stats_dict.get("games_played", 0)
                    stats.wins = stats_dict.get("wins", 0)
                    stats.losses = stats_dict.get("losses", 0)
                    stats.current_win_streak = stats_dict.get("current_win_streak", 0)
                    stats.best_win_streak = stats_dict.get("best_win_streak", 0)
                    stats.current_loss_streak = stats_dict.get("current_loss_streak", 0)
                    stats.worst_loss_streak = stats_dict.get("worst_loss_streak", 0)
                    stats.total_moves_made = stats_dict.get("total_moves_made", 0)
                    stats.avg_game_length = stats_dict.get("avg_game_length", 0.0)
                    stats.last_10_results = stats_dict.get("last_10_results", [])
            
            self.transition_history = state.get("transition_history", [])
            
            print(f"📚 Loaded curriculum: {self.current_phase.value} | "
                  f"{self.current_difficulty.value} | "
                  f"{self.total_games_played} games")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load curriculum state: {e}")
    
    def reset(self):
        """Reset the entire curriculum back to the beginning."""
        self.current_phase = TrainingPhase.PHASE_1_HUMAN_TEACHER
        self.current_difficulty = Difficulty.EASY
        self.total_games_played = 0
        self.total_human_rated_moves = 0
        self.difficulty_stats = {
            Difficulty.EASY.value: DifficultyStats(difficulty=Difficulty.EASY.value),
            Difficulty.MEDIUM.value: DifficultyStats(difficulty=Difficulty.MEDIUM.value),
            Difficulty.HARD.value: DifficultyStats(difficulty=Difficulty.HARD.value),
        }
        self.review_queue = []
        self.transition_history = []
        self._save_state()
        print("🔄 Curriculum reset to beginning!")


# ==============================================================================
# 🧪 DEMO — Test the curriculum system
# ==============================================================================

if __name__ == "__main__":
    print("📚 Curriculum Manager Demo")
    print("=" * 60)
    
    curriculum = CurriculumManager(save_path="demo_curriculum.json")
    print(curriculum.get_phase_description())
    
    import random
    print("\n🎮 Simulating 60 games on Easy difficulty...")
    for i in range(60):
        won = random.random() < 0.65
        moves = random.randint(20, 100)
        territory = random.uniform(-0.1, 0.5) if won else random.uniform(-0.3, 0.1)
        curriculum.record_game_result(won=won, moves_made=moves, territory_gained=territory)
        curriculum.record_human_rating(count=moves)
    
    print("\n" + curriculum.get_phase_description())
    
    progress = curriculum.get_promotion_progress()
    print(f"\n📈 Promotion Progress:")
    for key in ["games_played", "win_rate", "win_streak"]:
        p = progress[key]
        print(f"   {key}: {p['current']}/{p['needed']} ({p['progress']:.0%})")
    
    config = curriculum.get_training_config()
    print(f"\n⚙️ Current Training Config:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    if os.path.exists("demo_curriculum.json"):
        os.remove("demo_curriculum.json")
    
    print("\n✅ Demo complete!")
