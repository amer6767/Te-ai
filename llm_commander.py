"""
=============================================================================
llm_commander.py — Reflective LLM Commander for Territorial.io
=============================================================================

The "Aggressive General" — a reflective AI commander that:
  1. Remembers past games (via MemorySystem)
  2. Reads a 3×3 Grid report instead of a compass radar
  3. Outputs Zone-based commands (Zone 1-9) for free-cursor targeting
  4. Plays aggressively: never hesitates with >20k troops

Architecture:
  See → Remember → Plan → Act → Reflect

Designed to run on Kaggle T4/P100 (16GB VRAM) using unsloth pre-quantized
checkpoints that download in seconds instead of 30GB full-precision weights.

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

import re
import time
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from memory_system import MemorySystem

# ==============================================
# CONFIGURATION
# ==============================================

# Pre-quantized 4-bit models — download in seconds, fit 16GB VRAM
MODEL_CANDIDATES = [
    "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",  # Best reasoning (R1 distill)
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",    # Strong alternative
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",           # Another solid option
]

DEFAULT_MODEL_ID = MODEL_CANDIDATES[0]

# Generation parameters tuned for tactical decisions (not creative writing)
GENERATION_CONFIG = {
    "max_new_tokens": 200,       # Slightly more room for chain-of-thought
    "temperature": 0.4,          # Low = more deterministic tactics
    "top_p": 0.85,               # Nucleus sampling for slight variety
    "top_k": 40,                 # Limit vocabulary spread
    "repetition_penalty": 1.15,  # Prevent loops like "attack attack attack"
    "do_sample": True,           # Enable sampling (not pure greedy)
}

# Valid zone targets (1-9 grid zones + wait)
VALID_ZONES = {1, 2, 3, 4, 5, 6, 7, 8, 9}

# Zone aliases the LLM might hallucinate → map to zone IDs
ZONE_ALIASES = {
    "top-left": 1, "topleft": 1, "tl": 1, "top left": 1,
    "top-center": 2, "topcenter": 2, "tc": 2, "top center": 2, "top": 2,
    "top-right": 3, "topright": 3, "tr": 3, "top right": 3,
    "center-left": 4, "centerleft": 4, "cl": 4, "left": 4, "center left": 4,
    "center": 5, "middle": 5, "c": 5,
    "center-right": 6, "centerright": 6, "cr": 6, "right": 6, "center right": 6,
    "bot-left": 7, "botleft": 7, "bl": 7, "bottom-left": 7, "bottom left": 7,
    "bot-center": 8, "botcenter": 8, "bc": 8, "bottom": 8, "bottom-center": 8, "bottom center": 8,
    "bot-right": 9, "botright": 9, "br": 9, "bottom-right": 9, "bottom right": 9,
}

# Fallback command when the LLM output cannot be parsed
SAFE_FALLBACK = {"target_zone": 0, "slider_pct": 0}  # zone 0 = wait

# Logging
logger = logging.getLogger("llm_commander")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ==============================================
# LLM COMMANDER CLASS
# ==============================================

class LLMCommander:
    """
    Reflective LLM Commander with Memory, Grid Vision, and Aggression.

    Lifecycle:
        commander = LLMCommander()
        cmd = commander.decide(state, grid_text)
        # Returns {"target_zone": int (1-9 or 0=wait), "slider_pct": int}
        commander.record_outcome(state, action_str, terr_before, terr_after, troops_before, troops_after, grid_text)

    The model is loaded once and reused for the entire session.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto"):
        """
        Load the quantized model and tokenizer + initialize memory.

        Args:
            model_id: HuggingFace model identifier (must be a bnb-4bit variant)
            device:   "auto" puts layers on GPU first, overflow to CPU
        """
        logger.info(f"⏳ Loading LLM: {model_id}")
        load_start = time.time()

        # 4-bit quantization config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=self.bnb_config,
            device_map=device,
            torch_dtype=torch.float16,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        load_time = time.time() - load_start
        logger.info(f"✅ LLM loaded in {load_time:.1f}s | Device: {self.model.device}")

        # Initialize Memory System
        self.memory = MemorySystem()

        # Stats tracking
        self._decision_count = 0
        self._total_gen_time = 0.0
        self._parse_failures = 0

    # ==================================================
    # PROMPT BUILDER — The Aggressive General's Brain
    # ==================================================

    def build_prompt(self, game_state: dict, grid_text: str) -> str:
        """
        Build the reflective prompt with:
          1. SYSTEM — aggressive warlord personality + game rules
          2. MEMORY — recalled past experiences (learning)
          3. CONTEXT — live game state + grid report
          4. LOOP WARNING — if short-term memory detects repetition
          5. OUTPUT FORMAT — strict ZONE command format

        Args:
            game_state: dict from get_state() with troops, interest, time, etc.
            grid_text:  string from generate_grid_report()

        Returns:
            Fully formatted prompt string ready for tokenization
        """
        troops = game_state.get("troops", 0)
        interest = game_state.get("interest", 5.0)
        territory = game_state.get("percentage", 0)
        game_time = game_state.get("time", "0:00")
        red_interest = game_state.get("red_interest", False)
        cycle = game_state.get("cycle", 0)
        tick = game_state.get("tick", 0)

        # ---- System Instructions (The Aggressive General) ----
        system_msg = (
            "You are a ruthless, battle-hardened General commanding an army in Territorial.io. "
            "You learn from the past. You NEVER hesitate. You ALWAYS expand.\n\n"
            "CORE DOCTRINE:\n"
            "• You see the battlefield as a 3×3 Grid (9 Zones). Zone 5 is your base.\n"
            "• Neutral land (gray) is FREE — seize it immediately. Always prioritize it.\n"
            "• Enemy land (colored) costs troops — only attack weak or isolated enemies.\n"
            "• NEVER use a slider below 10%. Minimum aggression is 10%.\n"
            "• If your troops exceed 20,000: ATTACK NOW. Do not wait. Expand immediately.\n"
            "• If troops exceed 50,000: Use slider 40-60% to crush nearby enemies.\n"
            "• If Interest is 0.00% or RED INTEREST: STOP attacking. Output 'wait'.\n"
            "• Prefer zones marked 'Neutral (Free Land)' or 'Our Border + Neutral (Expand Here)'.\n"
            "• AVOID zones marked 'Ocean/Edge (Blocked)' — you cannot expand there.\n"
            "• AVOID zones marked 'Heavy Enemy Presence (Dangerous)' unless you have massive troops.\n"
            "• Consult the MEMORY section below. If a move worked before, repeat it. "
            "If it failed, try something different.\n"
            "• If the LOOP WARNING says you are repeating yourself, CHANGE your target zone.\n"
        )

        # ---- Memory Recall (The Journal) ----
        memory_text = self.memory.recall_experiences(game_state, grid_text)

        # ---- Loop Detection ----
        loop_warning = self.memory.detect_loop()
        loop_text = f"\n⚠️ LOOP WARNING: {loop_warning}\n" if loop_warning else ""

        # ---- Live Game Context ----
        red_warning = " ⚠️ RED INTEREST — DO NOT ATTACK!" if red_interest else ""
        context_msg = (
            f"CURRENT BATTLEFIELD:\n"
            f"  Time: {game_time} (Cycle {cycle}, Tick {tick})\n"
            f"  Territory: {territory:.1f}%\n"
            f"  Troops: {troops:,}\n"
            f"  Interest Rate: {interest:.2f}%{red_warning}\n\n"
            f"GRID SCAN (9 Zones):\n"
            f"  {grid_text}\n\n"
            f"{memory_text}\n"
            f"{loop_text}\n"
        )

        # ---- Output Format Instructions ----
        format_msg = (
            "Based on your doctrine, the grid scan, and your memory of past battles, "
            "decide your next move.\n"
            "Think briefly (1-2 sentences of reasoning), then output EXACTLY:\n\n"
            "COMMAND: TARGET=ZONE_X, SLIDER=Y\n\n"
            "Where X is a zone number (1-9) and Y is the slider percentage (10-100).\n"
            "To hold position and save troops: COMMAND: TARGET=WAIT, SLIDER=0\n"
        )

        # ---- Assemble using chat template ----
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": context_msg + format_msg},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"### System:\n{system_msg}\n\n### User:\n{context_msg}{format_msg}\n\n### Assistant:\n"

        return prompt

    # ==================================================
    # DECISION ENGINE — Generate + Parse
    # ==================================================

    def decide(self, game_state: dict, grid_text: str) -> dict:
        """
        The main entry point. Takes game state + grid, returns a command.

        Returns:
            {"target_zone": int (1-9 or 0=wait), "slider_pct": int}
        """
        self._decision_count += 1

        # Quick safety: red interest → don't even bother the LLM
        if game_state.get("red_interest", False):
            logger.info("🛑 Red interest detected — forcing wait (no LLM call)")
            return {"target_zone": 0, "slider_pct": 0}

        # Build prompt
        prompt = self.build_prompt(game_state, grid_text)

        # Generate
        gen_start = time.time()
        raw_output = self._generate(prompt)
        gen_time = time.time() - gen_start
        self._total_gen_time += gen_time

        logger.info(f"🧠 LLM generated in {gen_time:.2f}s | Output: {raw_output[:120]}...")

        # Parse
        command = self.parse_command(raw_output)

        # Aggression override: if troops > 20k and LLM said wait, force an attack
        troops = game_state.get("troops", 0)
        if command["target_zone"] == 0 and troops > 20000:
            # Find a neutral zone from the grid text to attack
            override_zone = self._find_best_zone_from_grid(grid_text)
            if override_zone:
                command["target_zone"] = override_zone
                command["slider_pct"] = max(command["slider_pct"], 25)
                logger.info(f"⚔️ Aggression override! Troops={troops:,}, forcing Zone {override_zone}")

        logger.info(
            f"📡 Decision #{self._decision_count}: "
            f"zone={command['target_zone']}, slider={command['slider_pct']}%"
        )

        return command

    def _generate(self, prompt: str) -> str:
        """Tokenize, generate, decode. Returns only new tokens."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **GENERATION_CONFIG,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, input_length:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return decoded.strip()

    # ==================================================
    # COMMAND PARSER — Extract Zone + Slider
    # ==================================================

    def parse_command(self, raw_output: str) -> dict:
        """
        Extract TARGET zone and SLIDER from the LLM's raw text output.

        Expected format:
            COMMAND: TARGET=ZONE_3, SLIDER=32

        Handles:
          - Case insensitivity
          - Zone name aliases (top-left → 1, etc.)
          - Bare numbers (TARGET=3)
          - WAIT as a valid target
          - Slider clamping to [0, 100]
          - Complete parse failures → safe fallback

        Returns:
            {"target_zone": int (1-9 or 0=wait), "slider_pct": int}
        """
        try:
            # --- Try strict format first ---
            # COMMAND: TARGET=ZONE_3, SLIDER=32
            pattern = r"COMMAND:\s*TARGET\s*=\s*(?:ZONE[_\s]?)?(\w+)\s*,\s*SLIDER\s*=\s*(\d+)"
            match = re.search(pattern, raw_output, re.IGNORECASE)

            if match:
                raw_zone = match.group(1).strip().lower()
                raw_slider = int(match.group(2))
            else:
                # --- Fallback: looser matching ---
                zone_match = re.search(
                    r"target\s*[=:]\s*(?:zone[_\s]?)?(\w+)", raw_output, re.IGNORECASE
                )
                slider_match = re.search(
                    r"slider\s*[=:]\s*(\d+)", raw_output, re.IGNORECASE
                )

                if zone_match and slider_match:
                    raw_zone = zone_match.group(1).strip().lower()
                    raw_slider = int(slider_match.group(1))
                else:
                    # --- Last resort: look for zone references ---
                    raw_zone = self._extract_zone_fuzzy(raw_output)
                    raw_slider = self._extract_slider_fuzzy(raw_output)

                    if raw_zone is None:
                        logger.warning(f"⚠️ Parse failure on: {raw_output[:200]}")
                        self._parse_failures += 1
                        return dict(SAFE_FALLBACK)

            # Normalize zone
            target_zone = self._normalize_zone(raw_zone)

            # Clamp slider
            slider_pct = max(0, min(100, raw_slider))

            # Enforce minimum aggression (never below 10% unless waiting)
            if target_zone > 0 and slider_pct < 10:
                slider_pct = 10

            # If waiting, slider must be 0
            if target_zone == 0:
                slider_pct = 0

            return {"target_zone": target_zone, "slider_pct": slider_pct}

        except Exception as e:
            logger.error(f"❌ Parse exception: {e} | Raw: {raw_output[:200]}")
            self._parse_failures += 1
            return dict(SAFE_FALLBACK)

    def _normalize_zone(self, raw: str) -> int:
        """Map raw zone string to zone ID (1-9) or 0 for wait."""
        raw = raw.strip().lower()

        # Wait keywords
        if raw in ("wait", "hold", "defend", "skip", "none", "0"):
            return 0

        # Direct numeric match
        try:
            zone_num = int(raw)
            if zone_num in VALID_ZONES:
                return zone_num
        except ValueError:
            pass

        # Alias match
        if raw in ZONE_ALIASES:
            return ZONE_ALIASES[raw]

        # Partial match
        for alias, zone_id in ZONE_ALIASES.items():
            if alias.startswith(raw) and len(raw) >= 3:
                return zone_id

        logger.warning(f"⚠️ Unknown zone '{raw}', defaulting to wait")
        return 0

    def _extract_zone_fuzzy(self, text: str) -> Optional[str]:
        """Last-resort fuzzy extraction: find any zone reference in text."""
        text_lower = text.lower()

        # Look for "zone N" pattern
        zone_match = re.search(r"zone\s*[_]?\s*(\d)", text_lower)
        if zone_match:
            return zone_match.group(1)

        # Look for zone name aliases
        for alias in sorted(ZONE_ALIASES.keys(), key=len, reverse=True):
            if alias in text_lower:
                return str(ZONE_ALIASES[alias])

        # Look for "wait" or "hold"
        for wait_word in ("wait", "hold", "defend", "skip"):
            if wait_word in text_lower:
                return "0"

        return None

    def _extract_slider_fuzzy(self, text: str) -> int:
        """Last-resort fuzzy extraction: find any reasonable number for slider."""
        numbers = re.findall(r"\b(\d{1,3})\b", text)
        for num_str in reversed(numbers):
            num = int(num_str)
            if 10 <= num <= 100:
                return num
        return 25  # safe default

    def _find_best_zone_from_grid(self, grid_text: str) -> Optional[int]:
        """Find the first zone with neutral land from the grid report."""
        grid_lower = grid_text.lower()
        # Priority: Free Land > Expand Here > Mixed Neutral
        for keyword in ["free land", "expand here", "mostly neutral", "neutral"]:
            for zone_id in range(1, 10):
                if zone_id == 5:
                    continue  # Skip our base zone
                zone_marker = f"zone {zone_id}"
                idx = grid_lower.find(zone_marker)
                if idx != -1:
                    snippet = grid_lower[idx:idx + 100]
                    if keyword in snippet:
                        return zone_id
        return None

    # ==================================================
    # OUTCOME RECORDING — Feed results back to memory
    # ==================================================

    def record_outcome(
        self,
        game_state: dict,
        action_str: str,
        territory_before: float,
        territory_after: float,
        troops_before: int,
        troops_after: int,
        grid_text: str = "",
    ):
        """
        Record the outcome of a step into the memory system.
        Called by run_session.py after each env.step().
        """
        self.memory.record_step(
            game_state=game_state,
            action=action_str,
            territory_before=territory_before,
            territory_after=territory_after,
            troops_before=troops_before,
            troops_after=troops_after,
            grid_text=grid_text,
        )

    def start_new_game(self):
        """Reset memory for a new game (short-term only)."""
        self.memory.start_new_game()

    def end_game(self, won: bool, final_territory: float):
        """Save all pending memories when a game ends."""
        self.memory.end_game(won=won, final_territory=final_territory)

    # ==================================================
    # STATS & DIAGNOSTICS
    # ==================================================

    def get_stats(self) -> dict:
        """Return performance statistics for logging/monitoring."""
        avg_time = (
            self._total_gen_time / self._decision_count
            if self._decision_count > 0 else 0.0
        )
        return {
            "decisions_made": self._decision_count,
            "total_gen_time_s": round(self._total_gen_time, 2),
            "avg_gen_time_s": round(avg_time, 2),
            "parse_failures": self._parse_failures,
            "parse_success_rate": (
                f"{(1 - self._parse_failures / max(1, self._decision_count)) * 100:.1f}%"
            ),
            "memory": self.memory.get_stats(),
        }


# ==============================================
# DEMO — Test locally without a game
# ==============================================

def demo():
    """
    Quick demo that loads the model and runs one decision cycle
    with synthetic game state and grid data.
    """
    print("\n🧠 LLM Commander Demo — Aggressive General Edition")
    print("=" * 55)

    # Synthetic game state
    fake_state = {
        "troops": 25000,
        "interest": 4.85,
        "percentage": 5.2,
        "time": "2:15",
        "red_interest": False,
        "cycle": 14,
        "tick": 5,
    }

    fake_grid = (
        "Zone 1 (Top-Left): Mostly Neutral (Free Land) | "
        "Zone 2 (Top-Center): Mixed Enemy + Neutral | "
        "Zone 3 (Top-Right): Ocean/Edge (Blocked) | "
        "Zone 4 (Center-Left): Our Border + Neutral (Expand Here) | "
        "Zone 5 (Center): Our Territory (Safe) | "
        "Zone 6 (Center-Right): Heavy Enemy Presence (Dangerous) | "
        "Zone 7 (Bot-Left): Mostly Neutral (Free Land) | "
        "Zone 8 (Bot-Center): Enemy Territory (Risky) | "
        "Zone 9 (Bot-Right): Ocean/Edge (Blocked)"
    )

    print("⏳ Loading model (this takes ~30s on first run)...")
    commander = LLMCommander()

    print("\n📡 Sending decision request...")
    command = commander.decide(fake_state, fake_grid)

    print(f"\n🎯 DECISION: {command}")
    print(f"\n📊 Stats: {commander.get_stats()}")
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
