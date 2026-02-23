"""
=============================================================================
rewards.py — Territorial.io 10-Factor Reward System
=============================================================================
Scores every move the AI makes on 10 different factors.

THE 10 FACTORS:
  1. Territory Change      — Did we gain or lose territory?
  2. Attack Timing         — Was this the right moment to attack?
  3. Target Selection      — Did we pick a good target?
  4. Border Efficiency     — Are our borders clean and defensible?
  5. Survival Instinct     — Did we avoid danger?
  6. Power Management      — Are we using power wisely?
  7. Opportunity Recognition — Did we spot and exploit openings?
  8. Multi-Enemy Awareness — Are we tracking ALL enemies?
  9. Map Position          — Is our position strategically strong?
  10. Aggression Balance   — Right amount of aggression?

Each factor is scored from -1.0 (terrible) to +1.0 (perfect).
Some factors are calculated automatically from game state.
Some need human input (returned as None for manual rating).

Also has win/loss bonus functions and a combined score calculator.

Works with: brain.py, decision.py, memory.py

=============================================================================
"""

# =============================================
# IMPORTS
# =============================================

import numpy as np
from collections import deque
import math

# Try to import project modules
try:
    from memory import FACTOR_NAMES, FACTOR_WEIGHTS, FACTOR_DESCRIPTIONS
except ImportError:
    FACTOR_NAMES = [
        "territory_change", "attack_timing", "target_selection",
        "border_efficiency", "survival_instinct", "power_management",
        "opportunity_recognition", "multi_enemy_awareness",
        "map_position", "aggression_balance",
    ]
    FACTOR_WEIGHTS = {name: 0.1 for name in FACTOR_NAMES}
    FACTOR_WEIGHTS["territory_change"] = 0.20
    FACTOR_WEIGHTS["survival_instinct"] = 0.15


# =============================================
# CONFIGURATION
# =============================================

class RewardConfig:
    """Settings for the reward system."""

    WIN_BONUS = 100.0
    LOSS_PENALTY = -50.0
    ELIMINATION_BONUS = 25.0
    SURVIVED_ROUND_BONUS = 1.0

    TERRITORY_GAIN_MULTIPLIER = 2.0
    TERRITORY_LOSS_MULTIPLIER = 3.0
    LARGE_GAIN_THRESHOLD = 5.0
    LARGE_LOSS_THRESHOLD = -3.0

    EARLY_GAME_MOVES = 20
    MID_GAME_MOVES = 60

    OPTIMAL_BORDER_RATIO = 0.3
    BORDER_RATIO_TOLERANCE = 0.1

    OPTIMAL_ATTACK_RATE = 0.6
    AGGRESSION_TOLERANCE = 0.15

    GRID_ROWS = 16
    GRID_COLS = 16


# =============================================
# FACTOR 1: TERRITORY CHANGE
# =============================================

def score_territory_change(prev_territory_pct, curr_territory_pct,
                            prev_screenshot=None, curr_screenshot=None):
    """
    Score Factor 1: Territory Change — Did we gain or lose territory?
    AUTO-SCORED: Yes — if we have territory percentages.
    Returns float from -1.0 to +1.0
    """
    if prev_territory_pct is None or curr_territory_pct is None:
        return None

    change = curr_territory_pct - prev_territory_pct

    if change >= RewardConfig.LARGE_GAIN_THRESHOLD:
        score = 1.0
    elif change > 0:
        score = min(0.8, change / RewardConfig.LARGE_GAIN_THRESHOLD * 0.8)
    elif change == 0:
        score = -0.05
    elif change > RewardConfig.LARGE_LOSS_THRESHOLD:
        score = max(-0.8, change / abs(RewardConfig.LARGE_LOSS_THRESHOLD) * 0.8)
    else:
        score = -1.0

    return max(-1.0, min(1.0, score))


# =============================================
# FACTOR 2: ATTACK TIMING
# =============================================

def score_attack_timing(action, move_number, total_moves_estimate=100,
                         territory_pct=None, enemy_territory_pct=None,
                         is_enemy_fighting=False):
    """
    Score Factor 2: Attack Timing — Was this the right moment?
    SEMI-AUTO: Uses heuristics based on game phase and state.
    """
    action_type = action.get("action_type", "unknown")

    if move_number <= RewardConfig.EARLY_GAME_MOVES:
        phase = "early"
    elif move_number <= RewardConfig.MID_GAME_MOVES:
        phase = "mid"
    else:
        phase = "late"

    score = 0.0

    if phase == "early":
        if action_type == "expand": score = 0.7
        elif action_type == "click": score = 0.0
        elif action_type == "wait": score = -0.3
        elif action_type == "defend": score = -0.1
    elif phase == "mid":
        if action_type == "click": score = 0.4
        elif action_type == "expand": score = 0.3
        elif action_type == "defend": score = 0.2
        elif action_type == "wait": score = -0.4
    elif phase == "late":
        if action_type == "click": score = 0.6
        elif action_type == "expand": score = 0.5
        elif action_type == "defend": score = 0.1
        elif action_type == "wait": score = -0.6

    if is_enemy_fighting and action_type == "click":
        score += 0.3

    if territory_pct is not None and territory_pct < 10:
        if action_type == "click":
            score -= 0.3

    if enemy_territory_pct is not None and territory_pct is not None:
        if enemy_territory_pct < territory_pct and action_type == "click":
            score += 0.2

    return max(-1.0, min(1.0, score))


# =============================================
# FACTOR 3: TARGET SELECTION
# =============================================

def score_target_selection(action, territory_pct=None, num_players=None,
                            target_is_weakest=None, target_is_neighbor=None):
    """Score Factor 3: Target Selection — Did we pick a good target?"""
    action_type = action.get("action_type", "unknown")
    if action_type != "click":
        return 0.0

    score = 0.0
    if target_is_weakest is True: score += 0.6
    if target_is_neighbor is True: score += 0.3
    elif target_is_neighbor is False: score -= 0.4
    if territory_pct is not None and territory_pct > 30: score += 0.1
    if num_players is not None and num_players <= 3: score += 0.2

    if target_is_weakest is None and target_is_neighbor is None:
        return None

    return max(-1.0, min(1.0, score))


# =============================================
# FACTOR 4: BORDER EFFICIENCY
# =============================================

def score_border_efficiency(our_border_tiles=None, our_total_tiles=None,
                             prev_border_ratio=None):
    """Score Factor 4: Border Efficiency — Are our borders clean?"""
    if our_border_tiles is None or our_total_tiles is None:
        return None
    if our_total_tiles == 0:
        return -1.0

    border_ratio = our_border_tiles / our_total_tiles
    optimal = RewardConfig.OPTIMAL_BORDER_RATIO
    tolerance = RewardConfig.BORDER_RATIO_TOLERANCE
    deviation = abs(border_ratio - optimal)

    if deviation <= tolerance:
        score = 0.8 - (deviation / tolerance) * 0.3
    elif deviation <= tolerance * 2:
        score = 0.2 - (deviation - tolerance) / tolerance * 0.4
    else:
        score = -0.2 - min(deviation * 2, 0.8)

    if prev_border_ratio is not None:
        if border_ratio < prev_border_ratio: score += 0.1
        elif border_ratio > prev_border_ratio + 0.05: score -= 0.1

    return max(-1.0, min(1.0, score))


# =============================================
# FACTOR 5: SURVIVAL INSTINCT
# =============================================

def score_survival_instinct(territory_pct, prev_territory_pct=None,
                              num_enemies_nearby=None, our_power=None,
                              enemy_power=None, game_over=False, we_died=False):
    """Score Factor 5: Survival Instinct — Did we stay safe?"""
    if we_died or (game_over and territory_pct is not None and territory_pct < 1):
        return -1.0
    if territory_pct is None:
        return None

    score = 0.0
    if territory_pct > 30: score += 0.5
    elif territory_pct > 15: score += 0.2
    elif territory_pct > 5: score -= 0.2
    else: score -= 0.6

    if prev_territory_pct is not None:
        change = territory_pct - prev_territory_pct
        if change < -5: score -= 0.3
        elif change < -2: score -= 0.1
        elif change > 0: score += 0.1

    if our_power is not None and enemy_power is not None:
        power_ratio = our_power / max(enemy_power, 0.01)
        if power_ratio > 1.5: score += 0.2
        elif power_ratio < 0.5: score -= 0.3

    if num_enemies_nearby is not None:
        if num_enemies_nearby == 0: score += 0.2
        elif num_enemies_nearby >= 3: score -= 0.2

    return max(-1.0, min(1.0, score))


# =============================================
# FACTOR 6: POWER MANAGEMENT
# =============================================

def score_power_management(action, power_before=None, power_after=None,
                            power_max=None, territory_pct=None):
    """Score Factor 6: Power Management — Using power wisely?"""
    if power_before is None:
        return None

    action_type = action.get("action_type", "unknown")
    score = 0.0
    power_pct = power_before / max(power_max, 1) * 100 if power_max else None

    if action_type == "click" and power_pct is not None:
        if power_pct < 10: score -= 0.7
        elif power_pct < 30: score -= 0.2
        elif power_pct > 70: score += 0.3
        else: score += 0.1

    if action_type in ["wait", "defend"] and power_pct is not None:
        if power_pct > 90: score -= 0.3
        elif power_pct < 30: score += 0.3

    return max(-1.0, min(1.0, score))


# =============================================
# FACTOR 7: OPPORTUNITY RECOGNITION
# =============================================

def score_opportunity_recognition(action, is_enemy_fighting=False,
                                    undefended_territory_nearby=False,
                                    enemy_just_lost=False,
                                    weak_enemy_nearby=False):
    """Score Factor 7: Opportunity Recognition — Did we spot openings?"""
    action_type = action.get("action_type", "unknown")
    score = 0.0
    has_info = False

    if is_enemy_fighting:
        has_info = True
        if action_type == "click": score += 0.8
        elif action_type == "expand": score += 0.5
        else: score -= 0.4

    if undefended_territory_nearby:
        has_info = True
        if action_type in ["click", "expand"]: score += 0.5
        elif action_type == "wait": score -= 0.5

    if enemy_just_lost:
        has_info = True
        if action_type == "click": score += 0.6
        elif action_type == "wait": score -= 0.3

    if weak_enemy_nearby:
        has_info = True
        if action_type == "click": score += 0.4
        elif action_type == "wait": score -= 0.2

    if not has_info:
        return None

    return max(-1.0, min(1.0, score))


# =============================================
# FACTOR 8: MULTI-ENEMY AWARENESS
# =============================================

def score_multi_enemy_awareness(num_players, enemies_tracked=None,
                                  got_backstabbed=False,
                                  watching_all_borders=False,
                                  action=None):
    """Score Factor 8: Multi-Enemy Awareness — Tracking ALL threats?"""
    if num_players is None:
        return None

    score = 0.0
    has_info = False

    if got_backstabbed:
        has_info = True
        score -= 0.9

    if watching_all_borders:
        has_info = True
        score += 0.6

    if enemies_tracked is not None:
        has_info = True
        actual_enemies = num_players - 1
        if actual_enemies > 0:
            tracking_pct = enemies_tracked / actual_enemies
            if tracking_pct >= 1.0: score += 0.4
            elif tracking_pct >= 0.5: score += 0.1
            else: score -= 0.3

    if num_players <= 2:
        score += 0.2

    if not has_info:
        return None

    return max(-1.0, min(1.0, score))


# =============================================
# FACTOR 9: MAP POSITION
# =============================================

def score_map_position(territory_center_row=None, territory_center_col=None,
                        territory_spread=None, has_corner=False,
                        has_edge=False, controls_chokepoint=False):
    """Score Factor 9: Map Position — Strong strategic position?"""
    score = 0.0
    has_info = False

    if territory_center_row is not None and territory_center_col is not None:
        has_info = True
        center_dist = math.sqrt(
            (territory_center_row - RewardConfig.GRID_ROWS / 2) ** 2 +
            (territory_center_col - RewardConfig.GRID_COLS / 2) ** 2
        )
        max_dist = math.sqrt((RewardConfig.GRID_ROWS / 2) ** 2 + (RewardConfig.GRID_COLS / 2) ** 2)
        center_score = 1.0 - (center_dist / max_dist)
        score += center_score * 0.3

    if has_corner:
        has_info = True
        score += 0.3

    if has_edge:
        has_info = True
        score += 0.15

    if controls_chokepoint:
        has_info = True
        score += 0.4

    if territory_spread is not None:
        has_info = True
        if territory_spread < 3: score += 0.2
        elif territory_spread < 5: score += 0.1
        elif territory_spread > 8: score -= 0.3
        else: score -= 0.1

    if not has_info:
        return None

    return max(-1.0, min(1.0, score))


# =============================================
# FACTOR 10: AGGRESSION BALANCE
# =============================================

def score_aggression_balance(attack_rate, territory_pct=None,
                              num_players=None, game_phase="mid"):
    """Score Factor 10: Aggression Balance — Right aggression level?"""
    if attack_rate is None:
        return None

    optimal = RewardConfig.OPTIMAL_ATTACK_RATE
    tolerance = RewardConfig.AGGRESSION_TOLERANCE

    if territory_pct is not None:
        if territory_pct < 15: optimal = 0.4
        elif territory_pct > 50: optimal = 0.7

    if game_phase == "early": optimal = 0.5
    elif game_phase == "late": optimal = 0.7

    if num_players is not None and num_players <= 2:
        optimal = 0.75

    deviation = abs(attack_rate - optimal)

    if deviation <= tolerance:
        score = 0.8 - (deviation / tolerance) * 0.3
    elif deviation <= tolerance * 2:
        score = 0.2 - (deviation - tolerance) / tolerance * 0.5
    elif attack_rate < optimal:
        score = -0.5 - min((optimal - attack_rate) * 2, 0.5)
    else:
        score = -0.3 - min((attack_rate - optimal) * 2, 0.7)

    return max(-1.0, min(1.0, score))


# =============================================
# WIN/LOSS BONUS FUNCTIONS
# =============================================

def win_loss_bonus(won, territory_pct=None, num_players_eliminated=0,
                    game_duration_moves=0, was_dominant=False):
    """Big bonus for winning, big penalty for losing."""
    if won:
        bonus = RewardConfig.WIN_BONUS
        if game_duration_moves > 0 and game_duration_moves < 50: bonus += 30.0
        elif game_duration_moves < 100: bonus += 15.0
        if was_dominant: bonus += 20.0
        if territory_pct is not None:
            if territory_pct > 90: bonus += 25.0
            elif territory_pct > 70: bonus += 15.0
            elif territory_pct > 50: bonus += 5.0
        bonus += num_players_eliminated * RewardConfig.ELIMINATION_BONUS
        return bonus
    else:
        penalty = RewardConfig.LOSS_PENALTY
        if game_duration_moves > 0 and game_duration_moves < 20: penalty -= 20.0
        elif game_duration_moves < 40: penalty -= 10.0
        if territory_pct is not None and territory_pct > 30: penalty += 15.0
        elif territory_pct is not None and territory_pct > 15: penalty += 5.0
        penalty += num_players_eliminated * 5.0
        return penalty


def per_move_survival_bonus():
    """Small bonus for surviving each move."""
    return RewardConfig.SURVIVED_ROUND_BONUS


def elimination_bonus(eliminated_player_territory_pct=None):
    """Bonus for eliminating another player."""
    base = RewardConfig.ELIMINATION_BONUS
    if eliminated_player_territory_pct is not None:
        if eliminated_player_territory_pct > 30: return base + 15.0
        elif eliminated_player_territory_pct > 15: return base + 5.0
    return base


# =============================================
# COMBINED SCORE CALCULATOR
# =============================================

def calculate_combined_score(factor_scores, custom_weights=None):
    """Combine all 10 factor scores into one overall move score."""
    weights = custom_weights or FACTOR_WEIGHTS
    total_weight = 0.0
    weighted_sum = 0.0
    rated_count = 0
    details = {}

    for name in FACTOR_NAMES:
        score = factor_scores.get(name)
        if score is not None:
            weight = weights.get(name, 0.1)
            contribution = score * weight
            weighted_sum += contribution
            total_weight += weight
            rated_count += 1
            details[name] = {"score": score, "weight": weight, "contribution": contribution}
        else:
            details[name] = {"score": None, "weight": weights.get(name, 0.1), "contribution": 0.0}

    combined = weighted_sum / total_weight if total_weight > 0 else None
    return combined, rated_count, details


# =============================================
# FULL MOVE SCORER
# =============================================

class MoveScorer:
    """Scores a complete move with all 10 factors."""

    def __init__(self):
        self.move_count = 0
        self.attack_count = 0
        self.total_score = 0.0

    def score_move(self, action, prev_territory_pct=None, curr_territory_pct=None,
                    move_number=0, territory_pct=None, num_players=None,
                    is_enemy_fighting=False, target_is_weakest=None,
                    target_is_neighbor=None, our_border_tiles=None,
                    our_total_tiles=None, our_power=None, enemy_power=None,
                    power_before=None, power_after=None, power_max=None,
                    undefended_nearby=False, enemy_just_lost=False,
                    weak_enemy_nearby=False, got_backstabbed=False,
                    territory_center=None, has_corner=False, has_edge=False,
                    game_over=False, we_died=False):
        """Score a move on all 10 factors using available data."""
        self.move_count += 1
        action_type = action.get("action_type", "unknown")
        if action_type in ["click", "expand"]:
            self.attack_count += 1

        attack_rate = self.attack_count / max(self.move_count, 1)

        if move_number <= RewardConfig.EARLY_GAME_MOVES: phase = "early"
        elif move_number <= RewardConfig.MID_GAME_MOVES: phase = "mid"
        else: phase = "late"

        factors = {
            "territory_change": score_territory_change(prev_territory_pct, curr_territory_pct),
            "attack_timing": score_attack_timing(action, move_number, territory_pct=territory_pct, is_enemy_fighting=is_enemy_fighting),
            "target_selection": score_target_selection(action, territory_pct=territory_pct, num_players=num_players, target_is_weakest=target_is_weakest, target_is_neighbor=target_is_neighbor),
            "border_efficiency": score_border_efficiency(our_border_tiles, our_total_tiles),
            "survival_instinct": score_survival_instinct(territory_pct or 0, prev_territory_pct, our_power=our_power, enemy_power=enemy_power, game_over=game_over, we_died=we_died),
            "power_management": score_power_management(action, power_before, power_after, power_max, territory_pct),
            "opportunity_recognition": score_opportunity_recognition(action, is_enemy_fighting, undefended_nearby, enemy_just_lost, weak_enemy_nearby),
            "multi_enemy_awareness": score_multi_enemy_awareness(num_players, got_backstabbed=got_backstabbed),
            "map_position": score_map_position(territory_center[0] if territory_center else None, territory_center[1] if territory_center else None, has_corner=has_corner, has_edge=has_edge),
            "aggression_balance": score_aggression_balance(attack_rate, territory_pct, num_players, phase),
        }

        combined, rated_count, details = calculate_combined_score(factors)
        if combined is not None:
            self.total_score += combined

        return {
            "factors": factors, "combined_score": combined, "rated_count": rated_count,
            "details": details, "attack_rate": attack_rate, "game_phase": phase,
            "avg_score": self.total_score / max(self.move_count, 1),
        }


# =============================================
# DEMO
# =============================================

def demo():
    import random
    print("\n" + "=" * 60)
    print("🏆 REWARD SYSTEM DEMO")
    print("=" * 60)

    scorer = MoveScorer()
    print("\n🎮 Scoring 10 simulated moves...\n")

    territory = 15.0
    for i in range(1, 11):
        action = {"action_type": random.choice(["click", "click", "expand", "wait", "defend"]), "action_index": random.randint(0, 258)}
        prev_territory = territory
        if action["action_type"] == "click": territory += random.uniform(-3, 5)
        elif action["action_type"] == "expand": territory += random.uniform(-1, 4)
        else: territory += random.uniform(-1, 1)
        territory = max(1, min(99, territory))

        result = scorer.score_move(action=action, prev_territory_pct=prev_territory, curr_territory_pct=territory,
            move_number=i * 5, territory_pct=territory, num_players=max(2, 8 - i // 2), is_enemy_fighting=random.random() < 0.3)

        score_str = f"{result['combined_score']:+.3f}" if result['combined_score'] is not None else "N/A"
        print(f"   Move {i:2d} | {action['action_type']:8s} | Territory: {territory:5.1f}% | Score: {score_str} | Rated: {result['rated_count']}/10")

    print(f"\n🏆 Win/Loss Examples:")
    print(f"   Quick dominant win:    +{win_loss_bonus(True, 95, 5, 30, True):.0f}")
    print(f"   Normal win:            +{win_loss_bonus(True, 60, 2, 80, False):.0f}")
    print(f"   Close loss:            {win_loss_bonus(False, 35, 1, 90, False):.0f}")
    print(f"   Early death:           {win_loss_bonus(False, 5, 0, 15, False):.0f}")

    print("\n✅ Reward system works!")
    return scorer


if __name__ == "__main__":
    demo()
