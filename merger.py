"""
=============================================================================
merger.py — Session Merger for Territorial.io AI
=============================================================================

Merges knowledge from all 6 training sessions (colab1-4, kaggle1-2)
into a single master model. Uses win-rate weighting so that better-
performing sessions contribute more to the merged knowledge.

The merger:
1. Reads all 6 session_*.json files
2. Weights each session by its win rate
3. Averages neural network weights proportionally
4. Keeps strategies that appear in 2+ sessions
5. Incorporates human feedback from rated_moves.json
6. Resolves conflicts by trusting higher win_rate sessions
7. Saves to master_model.json
8. Updates best_moves.json with moves rated good on 7+/10 factors

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

import json
import os
import time
import glob
import torch
import numpy as np
from typing import Dict, List, Optional
from collections import Counter, defaultdict


# ==============================================
# CONFIGURATION
# ==============================================

SESSION_NAMES = ["colab1", "colab2", "colab3", "colab4", "kaggle1", "kaggle2"]
SESSION_DIR = "sessions"
MODELS_DIR = "models"
MASTER_MODEL_FILE = "master_model.json"
BEST_MOVES_FILE = "best_moves.json"
RATED_MOVES_FILE = "rated_moves.json"
MIN_FACTORS_FOR_BEST = 7   # Need 7+/10 factors rated "good" for best_moves
MIN_SESSIONS_FOR_STRATEGY = 2  # Strategy must appear in 2+ sessions to keep


# ==============================================
# SESSION MERGER CLASS
# ==============================================

class SessionMerger:
    """
    Merges knowledge from all 6 training sessions into a master model.
    
    The merge uses win-rate weighting: sessions with higher win rates
    contribute more to the merged model parameters and strategy list.
    
    Usage:
        merger = SessionMerger()
        merger.merge_all()
        merger.print_merge_report()
    """

    def __init__(self):
        """
        Initialize the merger. Loads all 6 session files from local
        disk or defaults to empty sessions.
        """
        self.sessions: Dict[str, Dict] = {}
        self.merge_results: Dict = {}
        self._load_sessions()

    def _load_sessions(self):
        """Load all 6 session files from the sessions directory."""
        for name in SESSION_NAMES:
            # Try loading the canonical session file
            filepath = f"session_{name}.json"
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        self.sessions[name] = json.load(f)
                    print(f"   ✅ Loaded {filepath}")
                    continue
                except (json.JSONDecodeError, IOError) as e:
                    print(f"   ⚠️ Error loading {filepath}: {e}")

            # Try the sessions directory
            dir_path = os.path.join(SESSION_DIR, f"session_{name}.json")
            if os.path.exists(dir_path):
                try:
                    with open(dir_path, 'r') as f:
                        self.sessions[name] = json.load(f)
                    print(f"   ✅ Loaded {dir_path}")
                    continue
                except (json.JSONDecodeError, IOError) as e:
                    print(f"   ⚠️ Error loading {dir_path}: {e}")

            # Try finding any timestamped session file
            pattern = os.path.join(SESSION_DIR, f"session_{name}_*.json")
            matches = sorted(glob.glob(pattern))
            if matches:
                latest = matches[-1]
                try:
                    with open(latest, 'r') as f:
                        self.sessions[name] = json.load(f)
                    print(f"   ✅ Loaded {latest} (latest)")
                    continue
                except (json.JSONDecodeError, IOError) as e:
                    print(f"   ⚠️ Error loading {latest}: {e}")

            # Session not found — create empty placeholder
            self.sessions[name] = {
                "session_name": name,
                "total_games_played": 0,
                "win_rate": 0.0,
                "current_phase": "phase_1_human_teacher",
                "current_difficulty": "easy",
                "best_strategies": [],
                "mistakes_to_avoid": [],
                "factor_averages": {},
            }
            print(f"   ⚠️ No session file found for {name} — using empty defaults")

    def merge_all(self) -> Dict:
        """
        Perform the complete merge across all 6 sessions.
        
        Steps:
            1. Compute win-rate weights for each session
            2. Average neural network weights proportionally
            3. Keep strategies appearing in 2+ sessions
            4. Incorporate human feedback from rated_moves.json
            5. Resolve conflicts by trusting higher win_rate sessions
            6. Save to master_model.json
            7. Update best_moves.json
        
        Returns:
            Dictionary with merged model data
        """
        print("\n" + "=" * 60)
        print("🔄 MERGING ALL SESSIONS")
        print("=" * 60)

        # --- Step 1: Calculate win-rate weights ---
        weights = self._calculate_weights()
        print(f"\n📊 Session Weights (by win rate):")
        for name, weight in weights.items():
            win_rate = self._get_win_rate(name)
            games = self._get_games_played(name)
            print(f"   {name:10s}: weight={weight:.3f} "
                  f"(win_rate={win_rate:.1%}, games={games})")

        # --- Step 2: Average neural network weights ---
        merged_weights_path = self._merge_model_weights(weights)

        # --- Step 3: Keep common strategies ---
        top_strategies = self._merge_strategies()

        # --- Step 4: Collect patterns to avoid ---
        avoid_patterns = self._merge_avoid_patterns()

        # --- Step 5: Incorporate human feedback ---
        human_feedback = self._incorporate_human_feedback()

        # --- Step 6: Merge factor performance ---
        factor_performance = self._merge_factor_averages(weights)

        # --- Step 7: Determine overall stats ---
        total_games = sum(self._get_games_played(n) for n in SESSION_NAMES)
        overall_win_rate = sum(
            self._get_win_rate(n) * weights.get(n, 0)
            for n in SESSION_NAMES
        )

        # Find most advanced session for phase/difficulty
        most_advanced_phase = "phase_1_human_teacher"
        most_advanced_difficulty = "easy"
        for name in SESSION_NAMES:
            session = self.sessions.get(name, {})
            phase = self._extract_field(session, "current_phase", "phase_1_human_teacher")
            difficulty = self._extract_field(session, "current_difficulty", "easy")
            if phase == "phase_2_self_learning":
                most_advanced_phase = "phase_2_self_learning"
            diff_order = {"easy": 0, "medium": 1, "hard": 2}
            if diff_order.get(difficulty, 0) > diff_order.get(most_advanced_difficulty, 0):
                most_advanced_difficulty = difficulty

        # --- Build master model ---
        master = {
            "merged_at": time.time(),
            "contributing_sessions": [
                {
                    "session_name": name,
                    "win_rate": self._get_win_rate(name),
                    "games_played": self._get_games_played(name),
                    "weight": weights.get(name, 0),
                }
                for name in SESSION_NAMES
            ],
            "overall_win_rate": overall_win_rate,
            "current_phase": most_advanced_phase,
            "current_difficulty": most_advanced_difficulty,
            "model_weights_path": merged_weights_path or "models/master_model.pth",
            "top_strategies": top_strategies,
            "avoid_patterns": avoid_patterns,
            "factor_performance": factor_performance,
            "total_games_trained": total_games,
            "human_feedback_integrated": len(human_feedback),
        }

        # --- Save master model ---
        with open(MASTER_MODEL_FILE, 'w') as f:
            json.dump(master, f, indent=2)
        print(f"\n💾 Master model saved to {MASTER_MODEL_FILE}")

        # --- Update best moves ---
        self._update_best_moves()

        self.merge_results = master
        return master

    def _calculate_weights(self) -> Dict[str, float]:
        """
        Calculate contribution weight for each session based on win rate.
        Sessions with 0 games get 0 weight. Weights are normalized to sum to 1.
        """
        raw_weights = {}
        for name in SESSION_NAMES:
            win_rate = self._get_win_rate(name)
            games = self._get_games_played(name)
            # Weight = win_rate * sqrt(games) — reward both winning AND more data
            raw_weights[name] = win_rate * (games ** 0.5) if games > 0 else 0.0

        total = sum(raw_weights.values())
        if total == 0:
            # Equal weights if no data
            return {name: 1.0 / len(SESSION_NAMES) for name in SESSION_NAMES}

        return {name: w / total for name, w in raw_weights.items()}

    def _merge_model_weights(self, weights: Dict[str, float]) -> Optional[str]:
        """
        Average neural network .pth weights across sessions,
        weighted by each session's contribution weight.
        
        Returns the path to the saved merged model, or None if no models found.
        """
        model_states = {}
        for name in SESSION_NAMES:
            model_path = os.path.join(MODELS_DIR, f"model_{name}.pth")
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location="cpu")
                    model_states[name] = checkpoint.get("policy_net", checkpoint)
                except Exception as e:
                    print(f"   ⚠️ Could not load {model_path}: {e}")

        if not model_states:
            print("   ⚠️ No model files found — skipping weight merge")
            return None

        # Average the state dicts weighted by session performance
        merged_state = {}
        first_state = list(model_states.values())[0]

        for key in first_state.keys():
            # Weighted average of parameters
            weighted_sum = None
            total_weight = 0.0

            for name, state_dict in model_states.items():
                w = weights.get(name, 0)
                if key in state_dict:
                    param = state_dict[key].float()
                    if weighted_sum is None:
                        weighted_sum = param * w
                    else:
                        weighted_sum += param * w
                    total_weight += w

            if weighted_sum is not None and total_weight > 0:
                merged_state[key] = weighted_sum / total_weight

        # Save merged model
        os.makedirs(MODELS_DIR, exist_ok=True)
        merged_path = os.path.join(MODELS_DIR, "master_model.pth")
        torch.save({"policy_net": merged_state}, merged_path)
        print(f"   🧠 Merged model weights saved to {merged_path}")

        return merged_path

    def _merge_strategies(self) -> List[str]:
        """
        Keep strategies that appear in at least 2 sessions.
        """
        strategy_counts = Counter()
        for name in SESSION_NAMES:
            session = self.sessions.get(name, {})
            strategies = self._extract_field(session, "best_strategies", [])
            for strat in strategies:
                if isinstance(strat, str):
                    strategy_counts[strat] += 1
                elif isinstance(strat, dict) and "name" in strat:
                    strategy_counts[strat["name"]] += 1

        # Keep strategies in 2+ sessions
        top = [s for s, count in strategy_counts.most_common()
               if count >= MIN_SESSIONS_FOR_STRATEGY]
        print(f"   📋 Kept {len(top)} strategies (appearing in {MIN_SESSIONS_FOR_STRATEGY}+ sessions)")
        return top

    def _merge_avoid_patterns(self) -> List[str]:
        """
        Collect patterns to avoid from all sessions.
        Keep any pattern that appears in any session.
        """
        all_patterns = set()
        for name in SESSION_NAMES:
            session = self.sessions.get(name, {})
            patterns = self._extract_field(session, "mistakes_to_avoid", [])
            for p in patterns:
                if isinstance(p, str):
                    all_patterns.add(p)
                elif isinstance(p, dict) and "pattern" in p:
                    all_patterns.add(p["pattern"])
        return list(all_patterns)

    def _incorporate_human_feedback(self) -> List[Dict]:
        """
        Read rated_moves.json and incorporate human feedback into
        the merged model knowledge.
        """
        if not os.path.exists(RATED_MOVES_FILE):
            return []

        try:
            with open(RATED_MOVES_FILE, 'r') as f:
                rated = json.load(f)
            print(f"   👤 Incorporated {len(rated)} human-rated moves")
            return rated
        except (json.JSONDecodeError, IOError):
            return []

    def _merge_factor_averages(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Merge factor averages across sessions, weighted by win rate.
        """
        factor_names = [
            "territory_change", "attack_timing", "target_selection",
            "border_efficiency", "survival_instinct", "power_management",
            "opportunity_recognition", "multi_enemy_awareness",
            "map_position", "aggression_balance",
        ]

        merged = {}
        for factor in factor_names:
            weighted_sum = 0.0
            total_weight = 0.0
            for name in SESSION_NAMES:
                session = self.sessions.get(name, {})
                averages = self._extract_field(session, "factor_averages", {})
                if factor in averages:
                    w = weights.get(name, 0)
                    weighted_sum += averages[factor] * w
                    total_weight += w

            merged[factor] = weighted_sum / total_weight if total_weight > 0 else 0.0

        return merged

    def _update_best_moves(self):
        """
        Read rated_moves.json and extract moves where 7+ out of 10
        factors were rated "good" by the human. Save to best_moves.json.
        """
        if not os.path.exists(RATED_MOVES_FILE):
            print("   ⚠️ No rated_moves.json found — skipping best moves update")
            return

        try:
            with open(RATED_MOVES_FILE, 'r') as f:
                rated = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        best = []
        for move in rated:
            ratings = move.get("human_ratings", {})
            if not ratings:
                continue

            good_count = sum(1 for v in ratings.values() if v == "good")
            if good_count >= MIN_FACTORS_FOR_BEST:
                best_entry = {
                    "move_id": move.get("move_id", "unknown"),
                    "session": move.get("session", "unknown"),
                    "game_number": move.get("game_number", 0),
                    "action_type": move.get("action_type", "unknown"),
                    "game_situation": f"Game {move.get('game_number', 0)}, "
                                     f"move {move.get('move_number', 0)}",
                    "all_10_factor_ratings": ratings,
                    "why_great": f"Rated 'good' on {good_count}/10 factors by human reviewer",
                    "how_to_replicate": f"Action: {move.get('action_type', 'unknown')} "
                                        f"at grid ({move.get('grid_row', '?')}, "
                                        f"{move.get('grid_col', '?')})",
                }
                best.append(best_entry)

        with open(BEST_MOVES_FILE, 'w') as f:
            json.dump(best, f, indent=2)

        print(f"   ⭐ Updated {BEST_MOVES_FILE}: {len(best)} best moves")

    # --- Helper methods ---

    def _get_win_rate(self, name: str) -> float:
        """Extract win rate from a session, handling nested formats."""
        session = self.sessions.get(name, {})
        # Try top-level
        if "win_rate" in session:
            return float(session["win_rate"])
        # Try session_stats
        stats = session.get("session_stats", {})
        if "win_rate" in stats:
            return float(stats["win_rate"])
        return 0.0

    def _get_games_played(self, name: str) -> int:
        """Extract total games played from a session."""
        session = self.sessions.get(name, {})
        if "total_games_played" in session:
            return int(session["total_games_played"])
        stats = session.get("session_stats", {})
        if "games_played" in stats:
            return int(stats["games_played"])
        meta = session.get("metadata", {})
        if "total_games_ever" in meta:
            return int(meta["total_games_ever"])
        return 0

    def _extract_field(self, session: dict, field: str, default=None):
        """Extract a field from a session dict, checking multiple locations."""
        if field in session:
            return session[field]
        for key in ["session_stats", "metadata", "curriculum_state"]:
            sub = session.get(key, {})
            if field in sub:
                return sub[field]
        return default

    def print_merge_report(self):
        """Print a detailed report of the merge results."""
        if not self.merge_results:
            print("⚠️ No merge has been performed yet. Run merge_all() first.")
            return

        r = self.merge_results
        print(f"\n{'='*60}")
        print(f"📊 MERGE REPORT")
        print(f"{'='*60}")
        print(f"\n  Merged at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r.get('merged_at', 0)))}")
        print(f"  Total games trained: {r.get('total_games_trained', 0)}")
        print(f"  Overall win rate: {r.get('overall_win_rate', 0):.1%}")
        print(f"  Current phase: {r.get('current_phase', 'unknown')}")
        print(f"  Current difficulty: {r.get('current_difficulty', 'unknown')}")

        print(f"\n  📈 Session Contributions:")
        for session_info in r.get("contributing_sessions", []):
            pct = session_info.get("weight", 0) * 100
            print(f"     {session_info['session_name']:10s}: "
                  f"{pct:5.1f}% contribution "
                  f"(win_rate={session_info.get('win_rate', 0):.1%}, "
                  f"games={session_info.get('games_played', 0)})")

        print(f"\n  🎯 Top strategies kept: {len(r.get('top_strategies', []))}")
        for strat in r.get("top_strategies", [])[:5]:
            print(f"     • {strat}")

        print(f"\n  ⚠️ Patterns to avoid: {len(r.get('avoid_patterns', []))}")
        for pat in r.get("avoid_patterns", [])[:5]:
            print(f"     • {pat}")

        print(f"\n  👤 Human feedback integrated: {r.get('human_feedback_integrated', 0)} moves")
        print(f"\n{'='*60}")


# ==============================================
# DEMO
# ==============================================

def demo():
    """Quick demo — runs merge with whatever session data exists."""
    print("\n🔄 Session Merger Demo")
    print("=" * 50)

    merger = SessionMerger()
    result = merger.merge_all()
    merger.print_merge_report()

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
