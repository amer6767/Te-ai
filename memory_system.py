"""
=============================================================================
memory_system.py — Persistent Memory & Learning for Territorial.io AI
=============================================================================

The "Warlord's Journal" — a reflective memory system that lets the AI
learn from its own past games.

Architecture:
  1. Short-Term Memory: Rolling buffer of last N actions to prevent loops
  2. Long-Term Memory: JSON database of past (state → action → outcome)
  3. State Signatures: Fingerprint the current moment for similarity search
  4. Recall: Before each LLM call, retrieve the top-K most relevant past
     experiences and inject them into the prompt as plain English

The AI improves across games by remembering what worked and what killed it.

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

import json
import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from collections import deque

# ==============================================
# CONFIGURATION
# ==============================================

MEMORY_FILE = "long_term_memory.json"
MAX_LONG_TERM_ENTRIES = 500        # Cap the JSON file size
SHORT_TERM_SIZE = 10               # Rolling window of recent actions
TOP_K_RECALL = 3                   # How many past experiences to inject
SAVE_INTERVAL = 50                 # Save to disk every N steps

# Troop bracket boundaries for state fingerprinting
TROOP_BRACKETS = [
    (0,      "NoTroops"),
    (5000,   "LowTroops"),
    (15000,  "MedTroops"),
    (40000,  "HighTroops"),
    (100000, "MassiveTroops"),
]

# Game phase boundaries (total seconds)
PHASE_BRACKETS = [
    (0,   "Opening"),       # 0:00 – 0:59
    (60,  "EarlyGame"),     # 1:00 – 2:59
    (180, "MidGame"),       # 3:00 – 5:59
    (360, "LateGame"),      # 6:00+
]

logger = logging.getLogger("memory_system")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ==============================================
# DATA STRUCTURES
# ==============================================

@dataclass
class MemoryEntry:
    """
    A single remembered experience from a past game.

    Fields:
        state_signature: Fingerprint of the game moment
                         e.g. "HighTroops_MidGame_Zone2Enemy"
        action:          What command the AI executed
                         e.g. "zone_3_slider_45"
        result:          What happened afterward
                         e.g. "territory_gained_1.2pct"
        score:           Numeric quality rating (-10 to +10)
        timestamp:       Unix time when this was recorded
        game_id:         Which game session this came from
    """
    state_signature: str
    action: str
    result: str
    score: float
    timestamp: float = field(default_factory=time.time)
    game_id: int = 0


# ==============================================
# MEMORY SYSTEM CLASS
# ==============================================

class MemorySystem:
    """
    Persistent memory that survives across game sessions.

    Short-term: deque of last N actions (prevents loops within a game)
    Long-term:  JSON file of (state, action, outcome, score) entries

    Usage:
        memory = MemorySystem()
        memory.record_step(state, action, result, score)           # after each step
        context = memory.recall_experiences(current_game_state)     # before LLM call
        memory.save_experience()                                    # end of game
    """

    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        self.game_id = int(time.time())  # Unique ID for this game session

        # Short-term: rolling window of recent actions (prevents loops)
        self.short_term: deque = deque(maxlen=SHORT_TERM_SIZE)

        # Long-term: loaded from JSON on init, appended during play
        self.long_term: List[dict] = self._load_long_term()

        # Buffer of new entries to be saved at end of game
        self._pending_entries: List[MemoryEntry] = []

        # Step counter for periodic saves
        self._step_count = 0

        logger.info(
            f"🧠 Memory initialized | "
            f"Long-term entries: {len(self.long_term)} | "
            f"File: {self.memory_file}"
        )

    # ------------------------------------------
    # STATE SIGNATURE — Fingerprint the Moment
    # ------------------------------------------

    @staticmethod
    def generate_state_signature(game_state: dict, grid_text: str = "") -> str:
        """
        Create a simplified string describing the current game moment.

        Combines troop level, game phase, and dominant grid threats into
        a compact signature for similarity matching.

        Examples:
            "HighTroops_MidGame_EnemyZone2_EnemyZone6"
            "LowTroops_EarlyGame_NeutralDominant"
            "MassiveTroops_LateGame_Surrounded"

        Args:
            game_state: dict from get_state() with troops, time, etc.
            grid_text:  string from generate_grid_report()

        Returns:
            Human-readable state fingerprint string
        """
        troops = game_state.get("troops", 0)
        game_time = game_state.get("time", "0:00")
        territory = game_state.get("percentage", 0)

        # --- Troop bracket ---
        troop_label = "NoTroops"
        for threshold, label in reversed(TROOP_BRACKETS):
            if troops >= threshold:
                troop_label = label
                break

        # --- Game phase ---
        try:
            parts = game_time.split(":")
            total_sec = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else 0
        except (ValueError, IndexError):
            total_sec = 0

        phase_label = "Opening"
        for threshold, label in reversed(PHASE_BRACKETS):
            if total_sec >= threshold:
                phase_label = label
                break

        # --- Territory size ---
        if territory < 2:
            terr_label = "Tiny"
        elif territory < 8:
            terr_label = "Small"
        elif territory < 20:
            terr_label = "Medium"
        else:
            terr_label = "Large"

        # --- Grid threat analysis ---
        threat_tags = []
        if grid_text:
            grid_lower = grid_text.lower()
            for zone_num in range(1, 10):
                zone_marker = f"zone {zone_num}"
                # Find this zone's description in the grid text
                idx = grid_lower.find(zone_marker)
                if idx != -1:
                    # Grab a window of text after the zone marker
                    snippet = grid_lower[idx:idx + 80]
                    if "heavy enemy" in snippet or "enemy dominant" in snippet:
                        threat_tags.append(f"EnemyZone{zone_num}")
                    elif "mostly neutral" in snippet:
                        threat_tags.append(f"NeutralZone{zone_num}")

        # Cap threat tags to avoid overly long signatures
        threat_str = "_".join(threat_tags[:3]) if threat_tags else "NoThreats"

        signature = f"{troop_label}_{phase_label}_{terr_label}_{threat_str}"
        return signature

    # ------------------------------------------
    # RECORD — Save a Step to Short-Term + Buffer
    # ------------------------------------------

    def record_step(
        self,
        game_state: dict,
        action: str,
        territory_before: float,
        territory_after: float,
        troops_before: int,
        troops_after: int,
        grid_text: str = "",
    ):
        """
        Record one game step into short-term memory and the pending buffer.

        Called after every env.step() to track what happened.

        Args:
            game_state:       State dict at time of decision
            action:           String describing the action (e.g. "zone_3_slider_45")
            territory_before: Territory % before the action
            territory_after:  Territory % after the action
            troops_before:    Troop count before
            troops_after:     Troop count after
            grid_text:        Grid report at time of decision
        """
        self._step_count += 1

        # Calculate outcome
        terr_change = territory_after - territory_before
        troop_change = troops_after - troops_before

        # Score the move: territory gain is king, troop loss is bad
        score = terr_change * 50.0  # +50 per 1% territory gained
        if troop_change < 0:
            score += troop_change / 1000.0  # Penalize troop loss (mild)
        if troop_change > 0:
            score += 0.5  # Small bonus for troop growth

        # Build result description
        if terr_change > 0.1:
            result = f"territory_gained_{terr_change:.1f}pct"
        elif terr_change < -0.1:
            result = f"territory_lost_{abs(terr_change):.1f}pct"
        else:
            result = "no_change"

        if troop_change < -5000:
            result += "_heavy_troop_loss"
        elif troop_change > 2000:
            result += "_troops_grew"

        # State signature
        signature = self.generate_state_signature(game_state, grid_text)

        # Create entry
        entry = MemoryEntry(
            state_signature=signature,
            action=action,
            result=result,
            score=score,
            game_id=self.game_id,
        )

        # Add to short-term (loop detection)
        self.short_term.append({
            "action": action,
            "score": score,
            "step": self._step_count,
        })

        # Add to pending buffer (saved to disk later)
        self._pending_entries.append(entry)

        # Periodic save for safety
        if self._step_count % SAVE_INTERVAL == 0:
            self.save_experience()

    # ------------------------------------------
    # RECALL — Retrieve Relevant Past Experiences
    # ------------------------------------------

    def recall_experiences(self, game_state: dict, grid_text: str = "", top_k: int = TOP_K_RECALL) -> str:
        """
        The "Learning" engine. Searches long-term memory for situations
        similar to the current one and returns a plain-English summary
        that gets injected into the LLM prompt.

        Similarity is based on matching troop bracket and game phase.

        Args:
            game_state: Current state dict
            grid_text:  Current grid report
            top_k:      Number of past experiences to return

        Returns:
            Plain-English string for the LLM prompt, e.g.:
            "MEMORY: In 3 past games with HighTroops in MidGame:
             - Attacking Zone 3 resulted in territory gain (+1.2%).
             - Attacking Zone 6 resulted in heavy troop loss. AVOID."
        """
        if not self.long_term:
            return "MEMORY: No past experiences yet. Play aggressively to build your journal."

        current_sig = self.generate_state_signature(game_state, grid_text)
        sig_parts = current_sig.split("_")

        # Extract troop bracket and phase from current signature
        current_troops = sig_parts[0] if len(sig_parts) > 0 else ""
        current_phase = sig_parts[1] if len(sig_parts) > 1 else ""

        # Score each long-term entry by similarity
        scored = []
        for entry in self.long_term:
            entry_sig = entry.get("state_signature", "")
            entry_parts = entry_sig.split("_")
            entry_troops = entry_parts[0] if len(entry_parts) > 0 else ""
            entry_phase = entry_parts[1] if len(entry_parts) > 1 else ""

            # Similarity score: exact match on troop bracket + phase
            similarity = 0
            if entry_troops == current_troops:
                similarity += 2  # Same troop level is most important
            if entry_phase == current_phase:
                similarity += 1  # Same game phase

            if similarity > 0:
                scored.append((similarity, entry))

        if not scored:
            return "MEMORY: No similar past experiences found. Explore freely."

        # Sort by similarity (desc), then by score (desc) for tiebreaker
        scored.sort(key=lambda x: (x[0], x[1].get("score", 0)), reverse=True)
        top_entries = scored[:top_k]

        # Build plain-English summary
        lines = [f"MEMORY RECALL ({current_troops}, {current_phase}):"]
        for _, entry in top_entries:
            action = entry.get("action", "unknown")
            result = entry.get("result", "unknown")
            score = entry.get("score", 0)

            if score > 1.0:
                verdict = "✅ GOOD MOVE — repeat this"
            elif score < -1.0:
                verdict = "❌ BAD MOVE — avoid this"
            else:
                verdict = "⚪ Neutral outcome"

            lines.append(f"  • {action} → {result} ({verdict})")

        return "\n".join(lines)

    # ------------------------------------------
    # LOOP DETECTION — Check Short-Term Memory
    # ------------------------------------------

    def detect_loop(self) -> Optional[str]:
        """
        Check if the AI is stuck in a loop (repeating the same action).

        Returns:
            Warning string if loop detected, None otherwise.
        """
        if len(self.short_term) < 4:
            return None

        recent_actions = [entry["action"] for entry in self.short_term]
        last_3 = recent_actions[-3:]

        # If last 3 actions are identical, we're looping
        if len(set(last_3)) == 1 and last_3[0] != "wait":
            return (
                f"⚠️ LOOP DETECTED: You have done '{last_3[0]}' three times in a row. "
                f"Try a DIFFERENT zone or wait."
            )

        # If last 4 alternate between two actions, we're oscillating
        if len(recent_actions) >= 4:
            last_4 = recent_actions[-4:]
            if last_4[0] == last_4[2] and last_4[1] == last_4[3] and last_4[0] != last_4[1]:
                return (
                    f"⚠️ OSCILLATION DETECTED: You keep alternating between "
                    f"'{last_4[0]}' and '{last_4[1]}'. Pick ONE direction and commit."
                )

        return None

    # ------------------------------------------
    # SAVE — Persist to Disk
    # ------------------------------------------

    def save_experience(self):
        """
        Flush pending entries to the long-term JSON file.
        Called at end of game or every SAVE_INTERVAL steps.
        """
        if not self._pending_entries:
            return

        # Convert pending entries to dicts
        new_entries = [asdict(e) for e in self._pending_entries]

        # Merge with existing long-term memory
        self.long_term.extend(new_entries)

        # Trim to max size (keep most recent entries)
        if len(self.long_term) > MAX_LONG_TERM_ENTRIES:
            self.long_term = self.long_term[-MAX_LONG_TERM_ENTRIES:]

        # Write to disk
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.long_term, f, indent=2)
            logger.info(
                f"💾 Saved {len(new_entries)} new entries to memory "
                f"(total: {len(self.long_term)})"
            )
        except Exception as e:
            logger.error(f"❌ Failed to save memory: {e}")

        # Clear the pending buffer
        self._pending_entries.clear()

    def _load_long_term(self) -> List[dict]:
        """Load existing long-term memory from JSON file."""
        if not os.path.exists(self.memory_file):
            return []

        try:
            with open(self.memory_file, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                logger.info(f"📖 Loaded {len(data)} entries from {self.memory_file}")
                return data
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"⚠️ Could not load memory file: {e}")

        return []

    # ------------------------------------------
    # GAME LIFECYCLE
    # ------------------------------------------

    def start_new_game(self):
        """Reset short-term memory for a new game. Long-term persists."""
        self.game_id = int(time.time())
        self.short_term.clear()
        self._step_count = 0
        self._pending_entries.clear()
        logger.info(f"🆕 New game started (ID: {self.game_id})")

    def end_game(self, won: bool, final_territory: float):
        """
        Called when a game ends. Adjusts scores of all pending entries
        based on final outcome, then saves to disk.
        """
        # Bonus/penalty applied to all entries from this game
        outcome_bonus = 5.0 if won else -2.0

        for entry in self._pending_entries:
            entry.score += outcome_bonus

        # Add a summary entry
        summary = MemoryEntry(
            state_signature="GAME_SUMMARY",
            action=f"final_territory_{final_territory:.1f}pct",
            result="VICTORY" if won else "DEFEAT",
            score=10.0 if won else -5.0,
            game_id=self.game_id,
        )
        self._pending_entries.append(summary)

        # Flush everything to disk
        self.save_experience()
        logger.info(
            f"🏁 Game ended: {'VICTORY' if won else 'DEFEAT'} | "
            f"Territory: {final_territory:.1f}% | "
            f"Entries saved: {len(self._pending_entries)}"
        )

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            "long_term_entries": len(self.long_term),
            "pending_entries": len(self._pending_entries),
            "short_term_size": len(self.short_term),
            "steps_this_game": self._step_count,
            "game_id": self.game_id,
        }


# ==============================================
# DEMO — Test memory system standalone
# ==============================================

def demo():
    """
    Quick demo: create synthetic experiences, save, recall,
    and test loop detection.
    """
    print("\n🧠 Memory System Demo")
    print("=" * 55)

    # Clean up any old test file
    test_file = "demo_memory.json"
    if os.path.exists(test_file):
        os.remove(test_file)

    memory = MemorySystem(memory_file=test_file)

    # Simulate Game 1: attacking Zone 2 (Enemy) → bad result
    print("\n📝 Game 1: Attacking Zone 2 (Enemy)...")
    memory.start_new_game()

    fake_state = {"troops": 12000, "percentage": 3.0, "time": "1:30"}
    memory.record_step(
        game_state=fake_state,
        action="zone_2_slider_40",
        territory_before=3.0,
        territory_after=2.1,
        troops_before=12000,
        troops_after=3000,
        grid_text="Zone 2 (Top-Center): Heavy Enemy Presence",
    )
    memory.end_game(won=False, final_territory=2.1)

    # Simulate Game 2: attacking Zone 7 (Neutral) → good result
    print("\n📝 Game 2: Attacking Zone 7 (Neutral)...")
    memory.start_new_game()

    fake_state = {"troops": 15000, "percentage": 4.0, "time": "2:00"}
    memory.record_step(
        game_state=fake_state,
        action="zone_7_slider_25",
        territory_before=4.0,
        territory_after=6.5,
        troops_before=15000,
        troops_after=14000,
        grid_text="Zone 7 (Bot-Left): Mostly Neutral",
    )
    memory.end_game(won=True, final_territory=15.0)

    # Now test RECALL from a new game
    print("\n🔍 Game 3: Recalling experiences...")
    memory.start_new_game()
    new_state = {"troops": 13000, "percentage": 3.5, "time": "1:45"}
    recall = memory.recall_experiences(new_state, grid_text="Zone 2: Heavy Enemy")
    print(f"\n{recall}")

    # Test loop detection
    print("\n🔄 Testing loop detection...")
    for i in range(4):
        memory.short_term.append({"action": "zone_3_slider_30", "score": 0, "step": i})
    loop_warning = memory.detect_loop()
    print(f"  Loop warning: {loop_warning}")

    # Stats
    print(f"\n📊 Stats: {memory.get_stats()}")

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
