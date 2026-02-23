"""
=============================================================================
decision.py — Territorial.io Strategic Decision Engine
=============================================================================
This is the THINKING part of the AI. While brain.py "sees" the game,
decision.py "thinks" about what to do.

It takes the CNN output from brain.py and makes smart strategic decisions:
  - WHERE to attack (which territory border to push)
  - WHEN to attack vs defend vs wait
  - HOW aggressive to be based on game state
  - REMEMBERS last N actions for context (short-term memory)
  - EXPLORES randomly sometimes to discover new strategies
  - EXPLOITS learned knowledge to make winning moves

Works with brain.py — import both files and use together.

=============================================================================
"""

# =============================================
# IMPORTS
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import math
import time


# =============================================
# DECISION CONFIG
# =============================================

class DecisionConfig:
    """Configuration for the strategic decision engine."""

    SHORT_TERM_MEMORY_SIZE = 32
    LONG_TERM_MEMORY_SIZE = 5000
    GRID_ROWS = 16
    GRID_COLS = 16
    NUM_GRID_ACTIONS = GRID_ROWS * GRID_COLS
    TOTAL_ACTIONS = NUM_GRID_ACTIONS + 3

    EXPLORATION_START = 1.0
    EXPLORATION_END = 0.02
    EXPLORATION_DECAY = 0.9992

    UCB_EXPLORATION_WEIGHT = 2.0
    TEMPERATURE_START = 5.0
    TEMPERATURE_END = 0.1
    TEMPERATURE_DECAY = 0.9995

    LOOKAHEAD_DISCOUNT = 0.99
    REPEAT_PENALTY = 0.3
    MAX_REPEATS_BEFORE_PENALTY = 3

    ATTACK_CONFIDENCE_THRESHOLD = 0.6
    DEFEND_TERRITORY_THRESHOLD = 0.3
    EXPAND_TERRITORY_THRESHOLD = 0.5

    WINNING_STREAK_THRESHOLD = 3
    LOSING_STREAK_THRESHOLD = 3

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================
# SHORT-TERM MEMORY
# =============================================

class ShortTermMemory:
    """
    Tracks the last N actions taken by the AI.
    Gives the AI context — what it just did, streaks, attack heatmap.
    """

    def __init__(self, max_size=DecisionConfig.SHORT_TERM_MEMORY_SIZE):
        self.actions = deque(maxlen=max_size)
        self.action_counts = {}
        self.recent_rewards = deque(maxlen=max_size)
        self.attack_heatmap = np.zeros((DecisionConfig.GRID_ROWS, DecisionConfig.GRID_COLS))
        self.current_streak = 0
        self.streak_type = "neutral"

    def add_action(self, action, reward=0.0):
        """Record an action and its reward."""
        self.actions.append({
            "action": action,
            "reward": reward,
            "timestamp": time.time()
        })

        action_idx = action["action_index"]
        if len(self.actions) >= 2:
            prev_action = list(self.actions)[-2]["action"]["action_index"]
            if action_idx == prev_action:
                self.action_counts[action_idx] = self.action_counts.get(action_idx, 0) + 1
            else:
                self.action_counts[action_idx] = 1

        if action.get("action_type") == "click" and action.get("grid_row") is not None:
            row, col = action["grid_row"], action["grid_col"]
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = row + dr, col + dc
                    if 0 <= r < DecisionConfig.GRID_ROWS and 0 <= c < DecisionConfig.GRID_COLS:
                        distance = abs(dr) + abs(dc)
                        heat = max(0, 1.0 - distance * 0.25)
                        self.attack_heatmap[r][c] += heat

        self.attack_heatmap *= 0.95
        self.recent_rewards.append(reward)
        self._update_streak(reward)

    def _update_streak(self, reward):
        if reward > 0:
            self.current_streak = self.current_streak + 1 if self.current_streak > 0 else 1
        elif reward < 0:
            self.current_streak = self.current_streak - 1 if self.current_streak < 0 else -1

        if self.current_streak >= DecisionConfig.WINNING_STREAK_THRESHOLD:
            self.streak_type = "winning"
        elif self.current_streak <= -DecisionConfig.LOSING_STREAK_THRESHOLD:
            self.streak_type = "losing"
        else:
            self.streak_type = "neutral"

    def get_repeat_count(self, action_idx):
        return self.action_counts.get(action_idx, 0)

    def get_recent_actions(self, n=5):
        return list(self.actions)[-n:]

    def get_action_diversity(self):
        if len(self.actions) < 2:
            return 1.0
        recent = [a["action"]["action_index"] for a in list(self.actions)[-10:]]
        return len(set(recent)) / len(recent)

    def get_average_recent_reward(self, n=10):
        recent = list(self.recent_rewards)[-n:]
        return np.mean(recent) if recent else 0.0

    def get_most_attacked_region(self):
        if np.max(self.attack_heatmap) == 0:
            return None
        idx = np.unravel_index(np.argmax(self.attack_heatmap), self.attack_heatmap.shape)
        return (int(idx[0]), int(idx[1]))

    def get_least_attacked_region(self):
        idx = np.unravel_index(np.argmin(self.attack_heatmap), self.attack_heatmap.shape)
        return (int(idx[0]), int(idx[1]))

    def get_context_vector(self):
        context = np.zeros(16)
        context[0] = self.get_average_recent_reward()
        context[1] = self.get_action_diversity()
        context[2] = np.clip(self.current_streak / 10.0, -1.0, 1.0)
        context[3] = min(len(self.actions) / DecisionConfig.SHORT_TERM_MEMORY_SIZE, 1.0)

        recent = self.get_recent_actions(4)
        action_type_map = {"click": 0.5, "wait": -0.5, "defend": -0.3, "expand": 1.0}
        for i, act in enumerate(recent):
            context[4 + i] = action_type_map.get(act["action"]["action_type"], 0.0)

        hot_spot = self.get_most_attacked_region()
        if hot_spot:
            context[8] = hot_spot[0] / DecisionConfig.GRID_ROWS
            context[9] = hot_spot[1] / DecisionConfig.GRID_COLS

        cold_spot = self.get_least_attacked_region()
        context[10] = cold_spot[0] / DecisionConfig.GRID_ROWS
        context[11] = cold_spot[1] / DecisionConfig.GRID_COLS

        context[12] = min(np.max(self.attack_heatmap) / 10.0, 1.0)
        streak_map = {"winning": 1.0, "neutral": 0.0, "losing": -1.0}
        context[13] = streak_map[self.streak_type]

        if len(self.recent_rewards) > 1:
            context[14] = min(np.std(list(self.recent_rewards)) / 10.0, 1.0)
        context[15] = min(self.action_counts.get(0, 0) / 50.0, 1.0)

        return context


# =============================================
# STRATEGIC THINKER — Dueling DQN
# =============================================

class StrategicThinker(nn.Module):
    """
    Dueling DQN: separates state value from action advantages.
    Input: CNN features [128] + memory context [16] = [144]
    Output: Q-values [259]
    """

    def __init__(self):
        super(StrategicThinker, self).__init__()

        input_size = 128 + 16

        self.shared = nn.Sequential(
            nn.Linear(input_size, 384),
            nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(384, 384),
            nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(384, 256), nn.GELU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, DecisionConfig.TOTAL_ACTIONS)
        )

    def forward(self, cnn_features, memory_context):
        combined = torch.cat([cnn_features, memory_context], dim=1)
        shared_features = self.shared(combined)
        value = self.value_head(shared_features)
        advantage = self.advantage_head(shared_features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# =============================================
# EXPLORATION STRATEGIES
# =============================================

class ExplorationStrategy:
    """Multiple exploration strategies: epsilon-greedy, Boltzmann, UCB."""

    def __init__(self):
        self.epsilon = DecisionConfig.EXPLORATION_START
        self.temperature = DecisionConfig.TEMPERATURE_START
        self.action_visits = np.zeros(DecisionConfig.TOTAL_ACTIONS)
        self.total_visits = 0
        self.current_strategy = "epsilon_greedy"

    def select_action(self, q_values, memory, force_strategy=None):
        strategy = force_strategy or self.current_strategy
        if strategy == "epsilon_greedy":
            return self._epsilon_greedy(q_values, memory)
        elif strategy == "boltzmann":
            return self._boltzmann(q_values, memory)
        elif strategy == "ucb":
            return self._ucb(q_values, memory)
        else:
            return self._greedy(q_values)

    def _epsilon_greedy(self, q_values, memory):
        diversity = memory.get_action_diversity()
        effective_epsilon = self.epsilon * (1.0 + (1.0 - diversity) * 0.5)
        effective_epsilon = min(effective_epsilon, 1.0)

        if random.random() < effective_epsilon:
            if self.total_visits > 0 and random.random() < 0.5:
                action_idx = int(np.argmin(self.action_visits))
            else:
                action_idx = random.randint(0, DecisionConfig.TOTAL_ACTIONS - 1)
            return action_idx, True, "epsilon_greedy"
        else:
            adjusted_q = self._apply_repeat_penalty(q_values, memory)
            action_idx = adjusted_q.argmax().item()
            return action_idx, False, "epsilon_greedy"

    def _boltzmann(self, q_values, memory):
        adjusted_q = self._apply_repeat_penalty(q_values, memory)
        probs = F.softmax(adjusted_q / self.temperature, dim=0)
        action_idx = torch.multinomial(probs, 1).item()
        was_random = action_idx != adjusted_q.argmax().item()
        return action_idx, was_random, "boltzmann"

    def _ucb(self, q_values, memory):
        adjusted_q = self._apply_repeat_penalty(q_values, memory)
        q_min = adjusted_q.min()
        q_max = adjusted_q.max()
        q_range = q_max - q_min
        if q_range > 0:
            normalized_q = (adjusted_q - q_min) / q_range
        else:
            normalized_q = adjusted_q * 0 + 0.5

        ucb_bonus = np.zeros(DecisionConfig.TOTAL_ACTIONS)
        if self.total_visits > 0:
            for a in range(DecisionConfig.TOTAL_ACTIONS):
                if self.action_visits[a] == 0:
                    ucb_bonus[a] = float('inf')
                else:
                    ucb_bonus[a] = DecisionConfig.UCB_EXPLORATION_WEIGHT * math.sqrt(
                        math.log(self.total_visits) / self.action_visits[a]
                    )

        ucb_bonus_tensor = torch.tensor(ucb_bonus, dtype=torch.float32).to(DecisionConfig.DEVICE)
        ucb_scores = normalized_q + ucb_bonus_tensor
        action_idx = ucb_scores.argmax().item()
        was_random = action_idx != adjusted_q.argmax().item()
        return action_idx, was_random, "ucb"

    def _greedy(self, q_values):
        return q_values.argmax().item(), False, "greedy"

    def _apply_repeat_penalty(self, q_values, memory):
        adjusted = q_values.clone()
        for action_idx in range(DecisionConfig.TOTAL_ACTIONS):
            repeat_count = memory.get_repeat_count(action_idx)
            if repeat_count > DecisionConfig.MAX_REPEATS_BEFORE_PENALTY:
                excess = repeat_count - DecisionConfig.MAX_REPEATS_BEFORE_PENALTY
                adjusted[action_idx] -= DecisionConfig.REPEAT_PENALTY * excess
        return adjusted

    def update_visit_counts(self, action_idx):
        self.action_visits[action_idx] += 1
        self.total_visits += 1

    def decay(self):
        self.epsilon = max(DecisionConfig.EXPLORATION_END, self.epsilon * DecisionConfig.EXPLORATION_DECAY)
        self.temperature = max(DecisionConfig.TEMPERATURE_END, self.temperature * DecisionConfig.TEMPERATURE_DECAY)

        if self.epsilon > 0.5:
            self.current_strategy = "epsilon_greedy"
        elif self.epsilon > 0.1:
            self.current_strategy = "boltzmann"
        else:
            self.current_strategy = "ucb"


# =============================================
# STRATEGIC ADVISOR
# =============================================

class StrategicAdvisor:
    """Recommends high-level strategy: AGGRESSIVE, DEFENSIVE, etc."""

    def __init__(self):
        self.current_strategy = "OPPORTUNISTIC"
        self.strategy_history = deque(maxlen=50)
        self.territory_history = deque(maxlen=100)

    def recommend_strategy(self, memory, territory_pct=None, num_players=None):
        if territory_pct is None:
            territory_pct = 10.0
        if num_players is None:
            num_players = 5

        self.territory_history.append(territory_pct)

        territory_trend = 0.0
        if len(self.territory_history) >= 5:
            recent = list(self.territory_history)[-5:]
            territory_trend = recent[-1] - recent[0]

        streak = memory.streak_type
        diversity = memory.get_action_diversity()

        if territory_pct > DecisionConfig.EXPAND_TERRITORY_THRESHOLD * 100:
            if num_players <= 2:
                strategy, reason = "AGGRESSIVE", "Dominating with few opponents"
            else:
                strategy, reason = "OPPORTUNISTIC", "Large territory, many opponents"
        elif territory_pct < DecisionConfig.DEFEND_TERRITORY_THRESHOLD * 100:
            if streak == "losing":
                strategy, reason = "TURTLE", "Small territory + losing streak"
            elif territory_trend < -2:
                strategy, reason = "DEFENSIVE", "Territory shrinking fast"
            else:
                strategy, reason = "EXPANSIONIST", "Small territory, need growth"
        else:
            if streak == "winning" and territory_trend > 0:
                strategy, reason = "AGGRESSIVE", "Winning streak with growth"
            elif streak == "losing":
                strategy, reason = "DEFENSIVE", "Mid-game losing streak"
            elif diversity < 0.3:
                strategy, reason = "EXPANSIONIST", "Too repetitive, need variety"
            else:
                strategy, reason = "OPPORTUNISTIC", "Balanced, watching for openings"

        modifiers = self._get_strategy_modifiers(strategy)
        self.current_strategy = strategy
        self.strategy_history.append(strategy)

        return {
            "strategy": strategy, "reason": reason, "modifiers": modifiers,
            "territory_pct": territory_pct, "territory_trend": territory_trend, "streak": streak,
        }

    def _get_strategy_modifiers(self, strategy):
        strategies = {
            "AGGRESSIVE":    {"click_bias": 0.3, "wait_bias": -0.5, "defend_bias": -0.2, "expand_bias": 0.4, "aggression": 0.9, "risk_tolerance": 0.8},
            "DEFENSIVE":     {"click_bias": -0.1, "wait_bias": 0.2, "defend_bias": 0.5, "expand_bias": -0.3, "aggression": 0.2, "risk_tolerance": 0.2},
            "OPPORTUNISTIC": {"click_bias": 0.1, "wait_bias": 0.1, "defend_bias": 0.1, "expand_bias": 0.1, "aggression": 0.5, "risk_tolerance": 0.5},
            "EXPANSIONIST":  {"click_bias": 0.2, "wait_bias": -0.3, "defend_bias": -0.1, "expand_bias": 0.5, "aggression": 0.7, "risk_tolerance": 0.6},
            "TURTLE":        {"click_bias": -0.3, "wait_bias": 0.3, "defend_bias": 0.4, "expand_bias": -0.4, "aggression": 0.1, "risk_tolerance": 0.1},
        }
        return strategies.get(strategy, strategies["OPPORTUNISTIC"])


# =============================================
# MASTER DECISION ENGINE
# =============================================

class MasterDecisionEngine:
    """Combines CNN vision + strategic thinker + memory + exploration."""

    def __init__(self, vision_cnn=None):
        if vision_cnn is None:
            try:
                from brain import VisionCNN
                self.vision = VisionCNN().to(DecisionConfig.DEVICE)
            except ImportError:
                self.vision = self._create_fallback_cnn()
        else:
            self.vision = vision_cnn

        self.thinker = StrategicThinker().to(DecisionConfig.DEVICE)
        self.memory = ShortTermMemory()
        self.explorer = ExplorationStrategy()
        self.advisor = StrategicAdvisor()
        self.optimizer = torch.optim.Adam(
            list(self.thinker.parameters()) + list(self.vision.parameters()), lr=0.0003
        )
        self.replay_buffer = deque(maxlen=DecisionConfig.LONG_TERM_MEMORY_SIZE)
        self.total_decisions = 0
        self.total_rewards = 0

        thinker_params = sum(p.numel() for p in self.thinker.parameters())
        print(f"\n🧩 Master Decision Engine Ready!")
        print(f"   StrategicThinker parameters: {thinker_params:,}")

    def _create_fallback_cnn(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )

    def decide(self, screenshot_tensor, territory_pct=None, num_players=None):
        """Make a strategic decision given the current game state."""
        self.total_decisions += 1

        with torch.no_grad():
            self.vision.eval()
            cnn_features = self.vision(screenshot_tensor)

        memory_context = self.memory.get_context_vector()
        memory_tensor = torch.tensor(memory_context, dtype=torch.float32).unsqueeze(0).to(DecisionConfig.DEVICE)

        strategy_info = self.advisor.recommend_strategy(self.memory, territory_pct, num_players)

        with torch.no_grad():
            self.thinker.eval()
            q_values = self.thinker(cnn_features, memory_tensor).squeeze(0)

        modifiers = strategy_info["modifiers"]
        modified_q = q_values.clone()
        modified_q[:DecisionConfig.NUM_GRID_ACTIONS] += modifiers["click_bias"]
        modified_q[DecisionConfig.NUM_GRID_ACTIONS] += modifiers["wait_bias"]
        modified_q[DecisionConfig.NUM_GRID_ACTIONS + 1] += modifiers["defend_bias"]
        modified_q[DecisionConfig.NUM_GRID_ACTIONS + 2] += modifiers["expand_bias"]

        action_idx, was_random, strategy_used = self.explorer.select_action(modified_q, self.memory)
        self.explorer.update_visit_counts(action_idx)
        self.explorer.decay()

        probs = F.softmax(modified_q, dim=0)
        confidence = float(probs[action_idx].item())
        action = self._decode_action(action_idx)

        return {
            **action, "was_random": was_random, "confidence": confidence,
            "strategy": strategy_info["strategy"], "strategy_reason": strategy_info["reason"],
            "exploration_type": strategy_used, "streak": strategy_info["streak"],
            "epsilon": self.explorer.epsilon, "temperature": self.explorer.temperature,
            "decision_number": self.total_decisions,
        }

    def record_outcome(self, action, reward):
        self.memory.add_action(action, reward)
        self.total_rewards += reward

    def _decode_action(self, action_idx):
        result = {"action_index": action_idx, "grid_row": None, "grid_col": None, "screen_x": None, "screen_y": None}
        if action_idx < DecisionConfig.NUM_GRID_ACTIONS:
            row = action_idx // DecisionConfig.GRID_COLS
            col = action_idx % DecisionConfig.GRID_COLS
            result["action_type"] = "click"
            result["grid_row"] = row
            result["grid_col"] = col
            result["screen_x"] = (col + 0.5) / DecisionConfig.GRID_COLS
            result["screen_y"] = (row + 0.5) / DecisionConfig.GRID_ROWS
        elif action_idx == DecisionConfig.NUM_GRID_ACTIONS:
            result["action_type"] = "wait"
        elif action_idx == DecisionConfig.NUM_GRID_ACTIONS + 1:
            result["action_type"] = "defend"
        else:
            result["action_type"] = "expand"
        return result

    def get_status(self):
        return {
            "total_decisions": self.total_decisions,
            "total_rewards": self.total_rewards,
            "avg_reward": self.total_rewards / max(self.total_decisions, 1),
            "epsilon": self.explorer.epsilon,
            "strategy": self.advisor.current_strategy,
            "streak": self.memory.streak_type,
        }


# =============================================
# DEMO
# =============================================

def demo():
    print("\n" + "=" * 60)
    print("🧩 DECISION ENGINE DEMO")
    print("=" * 60)

    engine = MasterDecisionEngine()
    fake = torch.randn(1, 3, 128, 128).to(DecisionConfig.DEVICE)

    print("\n🎮 Making 10 test decisions...\n")
    for i in range(10):
        action = engine.decide(fake, territory_pct=10 + i * 5, num_players=max(2, 8 - i))
        reward = random.uniform(-5, 10)
        engine.record_outcome(action, reward)
        emoji = {"AGGRESSIVE": "⚔️", "DEFENSIVE": "🛡️", "OPPORTUNISTIC": "👀", "EXPANSIONIST": "🌍", "TURTLE": "🐢"}.get(action["strategy"], "❓")
        print(f"   #{i+1:2d}: {action['action_type']:8s} | {emoji} {action['strategy']:15s} | Random: {'Y' if action['was_random'] else 'N'} | Reward: {reward:+.1f}")

    print("\n✅ Decision engine is working!")
    return engine


if __name__ == "__main__":
    demo()
