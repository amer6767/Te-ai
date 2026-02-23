"""
=============================================================================
memory.py — Territorial.io Game Memory System
=============================================================================
Stores EVERYTHING that happened during the current game.

For each move, stores:
  - The game screenshot (as a tensor)
  - The action taken (where, what type)
  - All 10 rating factors:
      1. Territory Change     — Did we gain or lose territory?
      2. Attack Timing        — Was this the right moment to attack?
      3. Target Selection     — Did we pick a good target?
      4. Border Efficiency    — Are we maintaining clean, defensible borders?
      5. Survival Instinct    — Did we avoid dangerous situations?
      6. Power Management     — Are we using our power wisely?
      7. Opportunity Recognition — Did we spot and exploit openings?
      8. Multi-Enemy Awareness — Are we aware of ALL threats, not just one?
      9. Map Position         — Is our territory in a strong map position?
      10. Aggression Balance  — Are we being the right amount of aggressive?

Can clear memory for new games and export for saving to session files.

Works with: brain.py, decision.py, rewards.py

=============================================================================
"""

# =============================================
# IMPORTS
# =============================================

import numpy as np                    # Math and arrays
import time                           # Timestamps for each move
import json                           # For exporting memory to JSON files
import os                             # File system operations
from collections import OrderedDict   # Ordered dictionary for clean data
from datetime import datetime         # Human-readable timestamps
import copy                           # Deep copy for safe memory snapshots

# Try to import torch (optional — only needed if storing tensors)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not found — memory will store screenshots as numpy arrays")


# =============================================
# THE 10 RATING FACTORS
# =============================================
# These are the 10 dimensions we rate every single move on.
# Each factor is scored from -1.0 (terrible) to +1.0 (perfect)
# None means "not rated yet" (needs human input or more data)

FACTOR_NAMES = [
    "territory_change",        # Factor 0: Did we gain or lose land?
    "attack_timing",           # Factor 1: Was the timing right?
    "target_selection",        # Factor 2: Did we pick the right target?
    "border_efficiency",       # Factor 3: Are our borders clean?
    "survival_instinct",       # Factor 4: Did we stay safe?
    "power_management",        # Factor 5: Using power wisely?
    "opportunity_recognition", # Factor 6: Did we spot openings?
    "multi_enemy_awareness",   # Factor 7: Aware of all threats?
    "map_position",            # Factor 8: Good position on map?
    "aggression_balance",      # Factor 9: Right level of aggression?
]

# Human-readable descriptions for each factor
FACTOR_DESCRIPTIONS = {
    "territory_change":        "How much territory we gained or lost (positive = gained, negative = lost)",
    "attack_timing":           "Whether we attacked at the right moment (when enemy was weak, when we were strong)",
    "target_selection":        "Whether we picked the best target to attack (weakest enemy, most valuable territory)",
    "border_efficiency":       "How clean and defensible our borders are (fewer border tiles = more efficient)",
    "survival_instinct":       "Whether we avoided dangerous situations that could kill us (staying away from strong enemies)",
    "power_management":        "Whether we used our power/resources wisely (not wasting attacks, saving for big moves)",
    "opportunity_recognition": "Whether we spotted and exploited openings (enemy fighting each other, undefended territory)",
    "multi_enemy_awareness":   "Whether we tracked ALL enemies, not just the one we're fighting (avoiding backstabs)",
    "map_position":            "Whether our territory is in a strong position (center control, corner safety, chokepoints)",
    "aggression_balance":      "Whether we're being the right amount of aggressive (too passive = lose, too aggressive = die)",
}

# How much each factor contributes to the overall score
# These weights determine which factors matter most
FACTOR_WEIGHTS = {
    "territory_change":        0.20,   # 20% — Most important: are we growing?
    "attack_timing":           0.12,   # 12% — Timing is crucial
    "target_selection":        0.12,   # 12% — Picking right targets matters
    "border_efficiency":       0.08,   # 8%  — Clean borders help long-term
    "survival_instinct":       0.15,   # 15% — Staying alive is critical
    "power_management":        0.08,   # 8%  — Resource management
    "opportunity_recognition": 0.10,   # 10% — Spotting openings wins games
    "multi_enemy_awareness":   0.05,   # 5%  — Awareness prevents surprises
    "map_position":            0.05,   # 5%  — Position advantage
    "aggression_balance":      0.05,   # 5%  — Balance matters
}


# =============================================
# SINGLE MOVE RECORD
# =============================================
# One move = one screenshot + one action + 10 factor ratings

class MoveRecord:
    """
    Stores everything about a single move in the game.

    A move record contains:
    - WHEN: timestamp of when the move was made
    - WHAT: the action taken (click, wait, defend, expand)
    - WHERE: grid position if it was a click
    - SEE: the game screenshot at that moment
    - RATE: all 10 factor ratings for this move
    - META: extra info (territory %, players remaining, etc.)

    Each factor rating is:
      - A float from -1.0 (terrible) to +1.0 (perfect)
      - Or None if not rated yet (needs human input)
    """

    def __init__(self, move_number, action, screenshot=None, metadata=None):
        # --- Basic Info ---
        self.move_number = move_number
        self.timestamp = time.time()
        self.datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- Action Taken ---
        self.action = copy.deepcopy(action) if action else {}
        self.action_type = action.get("action_type", "unknown")
        self.grid_row = action.get("grid_row", None)
        self.grid_col = action.get("grid_col", None)
        self.was_random = action.get("was_random", False)
        self.confidence = action.get("confidence", 0.0)
        self.strategy = action.get("strategy", "unknown")

        # --- Screenshot ---
        self.screenshot = self._process_screenshot(screenshot)
        self.has_screenshot = screenshot is not None

        # --- The 10 Rating Factors ---
        self.factors = OrderedDict()
        for name in FACTOR_NAMES:
            self.factors[name] = None  # None = not rated yet

        # --- Metadata ---
        self.metadata = metadata or {}

        # --- Computed Scores ---
        self.overall_score = None
        self.reward = None

    def _process_screenshot(self, screenshot):
        if screenshot is None:
            return None
        if HAS_TORCH and isinstance(screenshot, torch.Tensor):
            return screenshot.detach().cpu()
        if isinstance(screenshot, np.ndarray):
            return screenshot
        try:
            from PIL import Image
            if isinstance(screenshot, Image.Image):
                return np.array(screenshot)
        except ImportError:
            pass
        return None

    def rate_factor(self, factor_name, score):
        """Rate a single factor for this move (-1.0 to +1.0)."""
        if factor_name not in FACTOR_NAMES:
            print(f"⚠️  Unknown factor: {factor_name}")
            return
        if score is not None:
            score = max(-1.0, min(1.0, float(score)))
        self.factors[factor_name] = score
        self._recalculate_overall_score()

    def rate_all_factors(self, ratings_dict):
        """Rate multiple factors at once."""
        for name, score in ratings_dict.items():
            if name in FACTOR_NAMES:
                self.factors[name] = max(-1.0, min(1.0, float(score))) if score is not None else None
        self._recalculate_overall_score()

    def _recalculate_overall_score(self):
        total_weight = 0.0
        weighted_sum = 0.0
        for name in FACTOR_NAMES:
            score = self.factors[name]
            if score is not None:
                weight = FACTOR_WEIGHTS[name]
                weighted_sum += score * weight
                total_weight += weight
        if total_weight > 0:
            self.overall_score = weighted_sum / total_weight
        else:
            self.overall_score = None

    def get_rated_count(self):
        return sum(1 for v in self.factors.values() if v is not None)

    def get_unrated_factors(self):
        return [name for name, score in self.factors.items() if score is None]

    def is_fully_rated(self):
        return self.get_rated_count() == 10

    def to_dict(self):
        return {
            "move_number": self.move_number,
            "timestamp": self.timestamp,
            "datetime": self.datetime_str,
            "action_type": self.action_type,
            "grid_row": self.grid_row,
            "grid_col": self.grid_col,
            "was_random": self.was_random,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "has_screenshot": self.has_screenshot,
            "factors": dict(self.factors),
            "overall_score": self.overall_score,
            "reward": self.reward,
            "rated_count": self.get_rated_count(),
            "fully_rated": self.is_fully_rated(),
            "metadata": self.metadata,
        }

    def __repr__(self):
        rated = self.get_rated_count()
        score_str = f"{self.overall_score:+.2f}" if self.overall_score is not None else "unrated"
        return (
            f"Move #{self.move_number} | {self.action_type:8s} | "
            f"Score: {score_str} | Rated: {rated}/10 | "
            f"Strategy: {self.strategy}"
        )


# =============================================
# GAME MEMORY — Stores All Moves in a Game
# =============================================

class GameMemory:
    """
    Stores ALL moves from the current game.
    Acts like a list of MoveRecords with extra features.
    """

    def __init__(self, game_id=None):
        self.game_id = game_id or f"game_{int(time.time())}"
        self.start_time = time.time()
        self.start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.moves = []
        self.move_count = 0

        self.game_over = False
        self.won = None
        self.final_territory = None
        self.end_time = None

        self._action_type_counts = {"click": 0, "wait": 0, "defend": 0, "expand": 0, "unknown": 0}
        self._strategy_counts = {}
        self._factor_sums = {name: 0.0 for name in FACTOR_NAMES}
        self._factor_counts = {name: 0 for name in FACTOR_NAMES}

        print(f"📝 New GameMemory created: {self.game_id}")

    def add_move(self, action, screenshot=None, metadata=None):
        """Record a new move."""
        self.move_count += 1
        move = MoveRecord(move_number=self.move_count, action=action, screenshot=screenshot, metadata=metadata)
        self.moves.append(move)

        action_type = move.action_type
        if action_type in self._action_type_counts:
            self._action_type_counts[action_type] += 1
        else:
            self._action_type_counts["unknown"] += 1

        strategy = move.strategy
        self._strategy_counts[strategy] = self._strategy_counts.get(strategy, 0) + 1
        return move

    def rate_move(self, move_number, factor_name, score):
        """Rate a specific factor for a specific move."""
        if move_number < 1 or move_number > len(self.moves):
            print(f"⚠️  Move #{move_number} doesn't exist")
            return
        move = self.moves[move_number - 1]
        old_score = move.factors.get(factor_name)
        move.rate_factor(factor_name, score)
        if score is not None:
            if old_score is not None:
                self._factor_sums[factor_name] -= old_score
            else:
                self._factor_counts[factor_name] += 1
            self._factor_sums[factor_name] += score

    def rate_last_move(self, factor_name, score):
        if self.moves:
            self.rate_move(len(self.moves), factor_name, score)

    def rate_last_move_all(self, ratings_dict):
        if self.moves:
            move = self.moves[-1]
            move.rate_all_factors(ratings_dict)
            for name, score in ratings_dict.items():
                if name in FACTOR_NAMES and score is not None:
                    self._factor_sums[name] += score
                    self._factor_counts[name] += 1

    def get_move(self, move_number):
        if 1 <= move_number <= len(self.moves):
            return self.moves[move_number - 1]
        return None

    def get_last_move(self):
        return self.moves[-1] if self.moves else None

    def get_last_n_moves(self, n=5):
        return self.moves[-n:]

    def get_moves_by_type(self, action_type):
        return [m for m in self.moves if m.action_type == action_type]

    def get_moves_by_strategy(self, strategy):
        return [m for m in self.moves if m.strategy == strategy]

    def get_best_moves(self, n=5):
        rated = [m for m in self.moves if m.overall_score is not None]
        rated.sort(key=lambda m: m.overall_score, reverse=True)
        return rated[:n]

    def get_worst_moves(self, n=5):
        rated = [m for m in self.moves if m.overall_score is not None]
        rated.sort(key=lambda m: m.overall_score)
        return rated[:n]

    def end_game(self, won, final_territory=None):
        self.game_over = True
        self.won = won
        self.final_territory = final_territory
        self.end_time = time.time()
        result_emoji = "🏆" if won else "💀"
        duration = self.end_time - self.start_time
        print(f"\n{result_emoji} Game Over! {'WON' if won else 'LOST'}")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Total moves: {self.move_count}")
        if final_territory is not None:
            print(f"   Final territory: {final_territory:.1f}%")

    def clear(self):
        """Clear all memory for a new game."""
        old_count = self.move_count
        self.moves = []
        self.move_count = 0
        self.game_over = False
        self.won = None
        self.final_territory = None
        self.end_time = None
        self._action_type_counts = {"click": 0, "wait": 0, "defend": 0, "expand": 0, "unknown": 0}
        self._strategy_counts = {}
        self._factor_sums = {name: 0.0 for name in FACTOR_NAMES}
        self._factor_counts = {name: 0 for name in FACTOR_NAMES}
        self.game_id = f"game_{int(time.time())}"
        self.start_time = time.time()
        self.start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🗑️  Memory cleared! (was {old_count} moves)")
        print(f"📝 New game started: {self.game_id}")

    def get_game_stats(self):
        duration = (self.end_time or time.time()) - self.start_time
        factor_averages = {}
        for name in FACTOR_NAMES:
            if self._factor_counts[name] > 0:
                factor_averages[name] = self._factor_sums[name] / self._factor_counts[name]
            else:
                factor_averages[name] = None
        rated_moves = [m for m in self.moves if m.overall_score is not None]
        avg_score = np.mean([m.overall_score for m in rated_moves]) if rated_moves else None
        best_factor = max(factor_averages.items(), key=lambda x: x[1] if x[1] is not None else -999)
        worst_factor = min(factor_averages.items(), key=lambda x: x[1] if x[1] is not None else 999)

        return {
            "game_id": self.game_id, "duration_seconds": duration, "total_moves": self.move_count,
            "game_over": self.game_over, "won": self.won, "final_territory": self.final_territory,
            "action_counts": dict(self._action_type_counts), "strategy_counts": dict(self._strategy_counts),
            "rated_moves": len(rated_moves), "unrated_moves": self.move_count - len(rated_moves),
            "average_score": float(avg_score) if avg_score is not None else None,
            "factor_averages": factor_averages,
            "best_factor": {"name": best_factor[0], "average": best_factor[1]},
            "worst_factor": {"name": worst_factor[0], "average": worst_factor[1]},
            "random_move_count": sum(1 for m in self.moves if m.was_random),
            "smart_move_count": sum(1 for m in self.moves if not m.was_random),
        }

    def print_stats(self):
        stats = self.get_game_stats()
        print(f"\n{'='*60}")
        print(f"📊 GAME STATISTICS — {stats['game_id']}")
        print(f"{'='*60}")
        print(f"   Duration:        {stats['duration_seconds']:.1f}s")
        print(f"   Total moves:     {stats['total_moves']}")
        if stats['won'] is not None:
            print(f"   Result:          {'🏆 WON' if stats['won'] else '💀 LOST'}")
        print(f"\n   Factor Averages:")
        for name, avg in stats['factor_averages'].items():
            if avg is not None:
                bar_len = int((avg + 1) * 10)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"     {name:25s}: {avg:+.3f} [{bar}]")
            else:
                print(f"     {name:25s}: not rated")
        print(f"{'='*60}")

    def export_to_dict(self):
        return {
            "game_id": self.game_id, "start_time": self.start_time,
            "start_datetime": self.start_datetime, "end_time": self.end_time,
            "game_over": self.game_over, "won": self.won,
            "final_territory": self.final_territory, "total_moves": self.move_count,
            "moves": [move.to_dict() for move in self.moves],
            "stats": self.get_game_stats(),
        }

    def export_to_json(self, filepath=None):
        if filepath is None:
            filepath = f"session_{self.game_id}.json"
        data = self.export_to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        file_size = os.path.getsize(filepath)
        print(f"💾 Memory exported to {filepath} ({file_size:,} bytes)")
        return filepath

    def export_screenshots_as_tensors(self):
        if not HAS_TORCH:
            return None
        screenshots = []
        for move in self.moves:
            if move.screenshot is not None:
                if isinstance(move.screenshot, torch.Tensor):
                    screenshots.append(move.screenshot)
                elif isinstance(move.screenshot, np.ndarray):
                    tensor = torch.from_numpy(move.screenshot).float()
                    if tensor.dim() == 3 and tensor.shape[2] == 3:
                        tensor = tensor.permute(2, 0, 1)
                    tensor = tensor / 255.0
                    screenshots.append(tensor.unsqueeze(0))
        if not screenshots:
            return None
        batch = torch.cat(screenshots, dim=0)
        print(f"📸 Exported {len(screenshots)} screenshots as tensor: {batch.shape}")
        return batch

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        return self.moves[idx]

    def __iter__(self):
        return iter(self.moves)

    def __repr__(self):
        status = "IN PROGRESS" if not self.game_over else ("WON 🏆" if self.won else "LOST 💀")
        return f"GameMemory({self.game_id}) | {self.move_count} moves | {status}"


# =============================================
# MULTI-GAME SESSION MEMORY
# =============================================

class SessionMemory:
    """Stores multiple games for cross-game learning."""

    def __init__(self, session_name=None):
        self.session_name = session_name or f"session_{int(time.time())}"
        self.games = []
        self.current_game = None
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        print(f"📁 Session started: {self.session_name}")

    def new_game(self):
        if self.current_game is not None and self.current_game.move_count > 0:
            self._archive_current_game()
        self.current_game = GameMemory()
        return self.current_game

    def _archive_current_game(self):
        if self.current_game:
            game_data = self.current_game.export_to_dict()
            self.games.append(game_data)
            self.total_games += 1
            if self.current_game.won is True:
                self.wins += 1
            elif self.current_game.won is False:
                self.losses += 1

    def end_current_game(self, won, final_territory=None):
        if self.current_game:
            self.current_game.end_game(won, final_territory)
            self._archive_current_game()

    def get_win_rate(self):
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0

    def get_factor_trends(self):
        trends = {name: [] for name in FACTOR_NAMES}
        for game in self.games:
            stats = game.get("stats", {})
            averages = stats.get("factor_averages", {})
            for name in FACTOR_NAMES:
                avg = averages.get(name)
                if avg is not None:
                    trends[name].append(avg)
        return trends

    def export_session(self, filepath=None):
        if filepath is None:
            filepath = f"{self.session_name}.json"
        if self.current_game and self.current_game.move_count > 0:
            self._archive_current_game()
        data = {
            "session_name": self.session_name, "total_games": self.total_games,
            "wins": self.wins, "losses": self.losses,
            "win_rate": self.get_win_rate(), "games": self.games,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"💾 Session saved: {filepath}")
        return filepath

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"📁 SESSION SUMMARY — {self.session_name}")
        print(f"{'='*60}")
        print(f"   Total games:  {self.total_games}")
        print(f"   Wins:         {self.wins} 🏆")
        print(f"   Losses:       {self.losses} 💀")
        print(f"   Win rate:     {self.get_win_rate():.1f}%")
        print(f"{'='*60}")


# =============================================
# DEMO
# =============================================

def demo():
    import random
    print("\n" + "=" * 60)
    print("📝 MEMORY SYSTEM DEMO")
    print("=" * 60)

    memory = GameMemory(game_id="demo_game_001")
    print("\n🎮 Simulating a game with 15 moves...\n")

    for i in range(1, 16):
        action = {
            "action_type": random.choice(["click", "click", "click", "wait", "defend", "expand"]),
            "action_index": random.randint(0, 258),
            "grid_row": random.randint(0, 15), "grid_col": random.randint(0, 15),
            "was_random": random.random() < 0.3, "confidence": random.random(),
            "strategy": random.choice(["AGGRESSIVE", "DEFENSIVE", "OPPORTUNISTIC"]),
        }
        screenshot = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        metadata = {"territory_pct": 10 + i * 3 + random.uniform(-5, 5), "num_players": max(2, 8 - i // 3)}
        move = memory.add_move(action, screenshot, metadata)

        ratings = {}
        for factor in FACTOR_NAMES:
            if random.random() < 0.7:
                ratings[factor] = random.uniform(-0.5, 1.0)
        move.rate_all_factors(ratings)
        print(f"   {move}")

    memory.end_game(won=True, final_territory=55.3)
    memory.print_stats()

    print("\n🏅 Top 3 Best Moves:")
    for move in memory.get_best_moves(3):
        print(f"   {move}")

    print("\n✅ Memory system works!")
    return memory


if __name__ == "__main__":
    demo()
