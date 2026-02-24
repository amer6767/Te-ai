"""
=============================================================================
llm_commander.py — LLM-Powered AI Commander for Territorial.io
=============================================================================

Replaces the entire RL pipeline (brain, decision, memory, rewards, trainer,
curriculum, merger, master, move_recorder, review_interface) with a single
LLM-based decision engine.

Architecture:
  1. Load an 8B-parameter Instruct model in 4-bit quantization (fits 16GB GPU)
  2. Build a structured prompt from live game state + radar report
  3. Generate a short chain-of-thought response ending with a strict command
  4. Parse the command into a direction + slider percentage for the game engine

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

# ==============================================
# CONFIGURATION
# ==============================================

# Pre-quantized 4-bit models — download in seconds, fit 16GB VRAM
# Ranked by reasoning quality for game-strategy tasks:
MODEL_CANDIDATES = [
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",   # Best overall reasoning
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",           # Strong alternative
]

DEFAULT_MODEL_ID = MODEL_CANDIDATES[0]

# Generation parameters tuned for tactical decisions (not creative writing)
GENERATION_CONFIG = {
    "max_new_tokens": 150,       # Short: just reasoning + command
    "temperature": 0.4,          # Low = more deterministic tactics
    "top_p": 0.85,               # Nucleus sampling for slight variety
    "top_k": 40,                 # Limit vocabulary spread
    "repetition_penalty": 1.15,  # Prevent loops like "attack attack attack"
    "do_sample": True,           # Enable sampling (not pure greedy)
}

# Valid direction names the LLM can output
VALID_DIRECTIONS = {
    "north", "north-east", "east", "south-east",
    "south", "south-west", "west", "north-west",
    "wait",  # Special: do nothing this tick
}

# Direction aliases the LLM might hallucinate → map to canonical names
DIRECTION_ALIASES = {
    "northeast": "north-east", "ne": "north-east",
    "southeast": "south-east", "se": "south-east",
    "southwest": "south-west", "sw": "south-west",
    "northwest": "north-west", "nw": "north-west",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "up": "north", "down": "south", "left": "west", "right": "east",
    "none": "wait", "hold": "wait", "defend": "wait", "skip": "wait",
}

# Fallback command when the LLM output cannot be parsed
SAFE_FALLBACK = {"direction": "wait", "slider_pct": 0}

# Logging
logger = logging.getLogger("llm_commander")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ==============================================
# LLM COMMANDER CLASS
# ==============================================

class LLMCommander:
    """
    Loads an 8B Instruct model in 4-bit, builds tactical prompts from
    game state + radar, and returns parsed attack commands.

    Lifecycle:
        commander = LLMCommander()           # loads model (~30s first time)
        cmd = commander.decide(state, radar)  # returns {"direction": ..., "slider_pct": ...}

    The model is loaded once and reused for the entire session.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto"):
        """
        Load the quantized model and tokenizer.

        Args:
            model_id: HuggingFace model identifier (must be a bnb-4bit variant)
            device:   "auto" puts layers on GPU first, overflow to CPU
        """
        logger.info(f"⏳ Loading LLM: {model_id}")
        load_start = time.time()

        # 4-bit quantization config — this is how we squeeze 8B params into 16GB
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",             # NormalFloat4 — best quality
            bnb_4bit_use_double_quant=True,         # Quantize the quantization constants too
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=self.bnb_config,
            device_map=device,
            torch_dtype=torch.float16,
        )

        # Ensure pad token is set (some models don't have one)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        load_time = time.time() - load_start
        logger.info(f"✅ LLM loaded in {load_time:.1f}s | Device: {self.model.device}")

        # Stats tracking
        self._decision_count = 0
        self._total_gen_time = 0.0
        self._parse_failures = 0

    # ==================================================
    # PROMPT BUILDER — The Brain's Instructions
    # ==================================================

    def build_prompt(self, game_state: dict, radar_text: str) -> str:
        """
        Build a structured chat prompt that the Instruct model can follow.

        The prompt has three sections:
          1. SYSTEM — rules of the game, what the AI must do
          2. CONTEXT — live numeric state (troops, interest, radar)
          3. OUTPUT FORMAT — strict command format the parser expects

        Args:
            game_state: dict from get_state() with troops, interest, time, etc.
            radar_text: string from ScreenCapture.generate_radar_report()

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

        # ---- System Instructions ----
        system_msg = (
            "You are an AI commander playing Territorial.io, a real-time strategy game. "
            "Your goal is to survive and expand your territory.\n\n"
            "RULES YOU MUST FOLLOW:\n"
            "• Neutral land (gray) is FREE to take — always prioritize expanding into it.\n"
            "• Enemy land (colored) requires combat and costs troops — only attack weak enemies.\n"
            "• Do NOT attack massive enemies or enemies in multiple directions at once.\n"
            "• If Interest is 0.00% or you are in RED INTEREST, do NOT attack. Output 'wait'.\n"
            "• Use a LOW slider (10-25%) to nibble neutral land cheaply.\n"
            "• Use a HIGHER slider (30-60%) only to attack weak enemies or rush critical land.\n"
            "• NEVER use slider above 70% unless you are about to die.\n"
            "• If all directions show Ocean/Edge, output 'wait' — there is nothing to attack.\n"
            "• Prefer 'Very Close' targets over 'Far' targets to minimize troop waste.\n"
        )

        # ---- Live Game Context ----
        red_warning = " ⚠️ RED INTEREST — DO NOT ATTACK!" if red_interest else ""
        context_msg = (
            f"CURRENT GAME STATE:\n"
            f"  Time: {game_time} (Cycle {cycle}, Tick {tick})\n"
            f"  Territory: {territory:.1f}%\n"
            f"  Troops: {troops:,}\n"
            f"  Interest Rate: {interest:.2f}%{red_warning}\n\n"
            f"RADAR SCAN (8 directions from your base):\n"
            f"  {radar_text}\n"
        )

        # ---- Output Format Instructions ----
        format_msg = (
            "\nBased on the radar and your current state, decide your next move.\n"
            "Think briefly (1-2 sentences), then output your decision in EXACTLY this format:\n\n"
            "COMMAND: DIRECTION=[direction], SLIDER=[number]\n\n"
            "Where [direction] is one of: North, North-East, East, South-East, "
            "South, South-West, West, North-West, or 'wait' to hold position.\n"
            "Where [number] is the slider percentage (10 to 100).\n"
            "If you choose to wait, use: COMMAND: DIRECTION=wait, SLIDER=0\n"
        )

        # ---- Assemble using chat template if available ----
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": context_msg + format_msg},
        ]

        # Use the model's native chat template for best results
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: manual concatenation for models without chat templates
            prompt = f"### System:\n{system_msg}\n\n### User:\n{context_msg}{format_msg}\n\n### Assistant:\n"

        return prompt

    # ==================================================
    # DECISION ENGINE — Generate + Parse
    # ==================================================

    def decide(self, game_state: dict, radar_text: str) -> dict:
        """
        The main entry point. Takes game state + radar, returns a command.

        This method:
          1. Builds the prompt
          2. Tokenizes and runs model.generate()
          3. Decodes the output
          4. Parses the COMMAND line
          5. Returns {"direction": str, "slider_pct": int}

        On any failure, returns SAFE_FALLBACK (wait, slider 0).

        Args:
            game_state: dict from get_state()
            radar_text: str from generate_radar_report()

        Returns:
            dict with "direction" (str) and "slider_pct" (int)
        """
        self._decision_count += 1

        # Quick safety check: if in red interest, don't even bother the LLM
        if game_state.get("red_interest", False):
            logger.info("🛑 Red interest detected — forcing wait (no LLM call)")
            return {"direction": "wait", "slider_pct": 0}

        # Build prompt
        prompt = self.build_prompt(game_state, radar_text)

        # Generate
        gen_start = time.time()
        raw_output = self._generate(prompt)
        gen_time = time.time() - gen_start
        self._total_gen_time += gen_time

        logger.info(f"🧠 LLM generated in {gen_time:.2f}s | Output: {raw_output[:120]}...")

        # Parse
        command = self.parse_command(raw_output)

        logger.info(
            f"📡 Decision #{self._decision_count}: "
            f"direction={command['direction']}, slider={command['slider_pct']}%"
        )

        return command

    def _generate(self, prompt: str) -> str:
        """
        Tokenize the prompt, run model.generate(), decode the output.

        Returns only the NEW tokens (strips the input prompt from output).

        Args:
            prompt: Full formatted prompt string

        Returns:
            Decoded string of the model's generated response
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,   # Limit context to avoid OOM on long games
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **GENERATION_CONFIG,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Strip input tokens — only decode the new generated tokens
        new_tokens = output_ids[0, input_length:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return decoded.strip()

    # ==================================================
    # COMMAND PARSER — Extract Direction + Slider
    # ==================================================

    def parse_command(self, raw_output: str) -> dict:
        """
        Extract DIRECTION and SLIDER from the LLM's raw text output.

        Expected format somewhere in the output:
            COMMAND: DIRECTION=North, SLIDER=32

        Handles:
          - Case insensitivity
          - Direction aliases (NE → North-East, hold → wait)
          - Slider clamping to [0, 100]
          - Complete parse failures → safe fallback

        Args:
            raw_output: Raw decoded string from the LLM

        Returns:
            {"direction": str, "slider_pct": int}
        """
        try:
            # Primary regex: strict format
            pattern = r"COMMAND:\s*DIRECTION\s*=\s*([A-Za-z\-]+)\s*,\s*SLIDER\s*=\s*(\d+)"
            match = re.search(pattern, raw_output, re.IGNORECASE)

            if match:
                raw_direction = match.group(1).strip().lower()
                raw_slider = int(match.group(2))
            else:
                # Fallback regex: looser matching for slightly malformed output
                dir_match = re.search(
                    r"direction\s*[=:]\s*([A-Za-z\-]+)", raw_output, re.IGNORECASE
                )
                slider_match = re.search(
                    r"slider\s*[=:]\s*(\d+)", raw_output, re.IGNORECASE
                )

                if dir_match and slider_match:
                    raw_direction = dir_match.group(1).strip().lower()
                    raw_slider = int(slider_match.group(1))
                else:
                    # Last resort: look for any direction keyword in the text
                    raw_direction = self._extract_direction_fuzzy(raw_output)
                    raw_slider = self._extract_slider_fuzzy(raw_output)

                    if raw_direction is None:
                        logger.warning(f"⚠️ Parse failure on: {raw_output[:200]}")
                        self._parse_failures += 1
                        return dict(SAFE_FALLBACK)

            # Normalize direction
            direction = self._normalize_direction(raw_direction)

            # Clamp slider
            slider_pct = max(0, min(100, raw_slider))

            # Sanity: if direction is "wait", slider must be 0
            if direction == "wait":
                slider_pct = 0

            return {"direction": direction, "slider_pct": slider_pct}

        except Exception as e:
            logger.error(f"❌ Parse exception: {e} | Raw: {raw_output[:200]}")
            self._parse_failures += 1
            return dict(SAFE_FALLBACK)

    def _normalize_direction(self, raw: str) -> str:
        """Map raw direction string to canonical direction name."""
        raw = raw.strip().lower()

        # Direct match
        if raw in VALID_DIRECTIONS:
            return raw

        # Alias match
        if raw in DIRECTION_ALIASES:
            return DIRECTION_ALIASES[raw]

        # Partial match (e.g., "north-ea" → "north-east")
        for valid in VALID_DIRECTIONS:
            if valid.startswith(raw) and len(raw) >= 3:
                return valid

        logger.warning(f"⚠️ Unknown direction '{raw}', defaulting to 'wait'")
        return "wait"

    def _extract_direction_fuzzy(self, text: str) -> Optional[str]:
        """Last-resort fuzzy extraction: find any direction keyword in text."""
        text_lower = text.lower()
        # Check longest names first to avoid partial matches
        for direction in sorted(VALID_DIRECTIONS, key=len, reverse=True):
            if direction in text_lower:
                return direction
        for alias, canonical in DIRECTION_ALIASES.items():
            if alias in text_lower:
                return canonical
        return None

    def _extract_slider_fuzzy(self, text: str) -> int:
        """Last-resort fuzzy extraction: find any reasonable number for slider."""
        numbers = re.findall(r"\b(\d{1,3})\b", text)
        for num_str in reversed(numbers):  # prefer later numbers (closer to command)
            num = int(num_str)
            if 5 <= num <= 100:
                return num
        return 25  # safe default

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
        }


# ==============================================
# DEMO — Test locally without a game
# ==============================================

def demo():
    """
    Quick demo that loads the model and runs one decision cycle
    with synthetic game state and radar data.
    """
    print("\n🧠 LLM Commander Demo")
    print("=" * 55)

    # Synthetic game state
    fake_state = {
        "troops": 12500,
        "interest": 4.85,
        "percentage": 3.2,
        "time": "1:15",
        "red_interest": False,
        "cycle": 8,
        "tick": 3,
    }

    fake_radar = (
        "North: Ocean/Edge | North-East: Neutral (Very Close) | "
        "East: Enemy (Medium) | South-East: Neutral (Medium) | "
        "South: Enemy (Far) | South-West: Neutral (Very Close) | "
        "West: Own Territory (Very Close) | North-West: Ocean/Edge"
    )

    print("⏳ Loading model (this takes ~30s on first run)...")
    commander = LLMCommander()

    print("\n📡 Sending decision request...")
    command = commander.decide(fake_state, fake_radar)

    print(f"\n🎯 DECISION: {command}")
    print(f"\n📊 Stats: {commander.get_stats()}")
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
