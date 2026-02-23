"""
=============================================================================
action_controller.py — Territorial.io Browser Action Controller
=============================================================================

Translates AI decisions from decision.py/brain.py into actual browser
interactions via Playwright.

Actions supported:
  - "click":  Set slider + click at translated screen coordinates
  - "expand": Set slider to 32% + click nearest unclaimed area
  - "defend": Set slider to 10% + wait
  - "wait":   Simply wait, do nothing

Respects Greenbiscuit opening strategy for cycles 1-5.
Includes safety stop: if territory drops >15% in one step, pause 2 seconds.

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

from playwright.async_api import Page
from game_environment import get_state, set_slider, OPENING, EXPAND_PCT
import random
import time

# ==============================================
# CONFIGURATION
# ==============================================

# Default viewport dimensions
DEFAULT_VIEWPORT_WIDTH = 1280
DEFAULT_VIEWPORT_HEIGHT = 900

# Game area mapping — centered around spawn point (640, 450)
# Used to translate normalized (0-1) coordinates to pixel coordinates
GAME_AREA_X_CENTER = 640
GAME_AREA_Y_CENTER = 450
GAME_AREA_X_RADIUS = 400   # ±400 pixels from center
GAME_AREA_Y_RADIUS = 300   # ±300 pixels from center

# Safety stop threshold: if territory drops this much, pause
SAFETY_DROP_THRESHOLD = 0.15   # 15% territory drop in one step
SAFETY_PAUSE_MS = 2000         # 2 second pause on safety stop

# Low slider for defense
DEFEND_SLIDER_PCT = 10

# Step delay (ms) — how long to wait after each action
DEFAULT_STEP_DELAY_MS = 800


# ==============================================
# ACTION CONTROLLER CLASS
# ==============================================

class ActionController:
    """
    Executes AI actions in the Territorial.io browser game.
    
    Takes action dictionaries from brain.py's GameAgent and translates
    them into Playwright mouse clicks and slider adjustments.
    
    Features:
        - Greenbiscuit opening override for cycles 1-5
        - Coordinate translation (normalized → pixel)
        - Action verification (did the action have an effect?)
        - Safety stop on large territory drops
    
    Usage:
        controller = ActionController(page)
        await controller.execute(action_dict, cycle=3)
    """

    def __init__(self, page: Page, viewport_width: int = DEFAULT_VIEWPORT_WIDTH,
                 viewport_height: int = DEFAULT_VIEWPORT_HEIGHT):
        """
        Initialize the action controller.
        
        Args:
            page:            Playwright Page object
            viewport_width:  Browser viewport width in pixels
            viewport_height: Browser viewport height in pixels
        """
        self.page = page
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.prev_territory = 0.0
        self.action_count = 0
        self.safety_stops = 0

    async def execute(self, action_dict: dict, cycle: int = 1) -> dict:
        """
        Execute an action in the browser game.
        
        Handles all action types: click, expand, defend, wait.
        Automatically applies Greenbiscuit opening for cycles 1-5.
        
        Args:
            action_dict: Action dictionary from GameAgent with keys:
                action_type: "click", "expand", "defend", "wait"
                screen_x:    Normalized X coordinate (0.0 to 1.0)
                screen_y:    Normalized Y coordinate (0.0 to 1.0)
            cycle: Current game cycle number (1-indexed)
        
        Returns:
            Dictionary with execution details:
                executed:    True if action was performed
                action_type: The action type that was executed
                pixel_x:     Pixel X coordinate (if applicable)
                pixel_y:     Pixel Y coordinate (if applicable)
                slider_pct:  Slider value that was set
                safety_stop: True if safety stop was triggered
        """
        self.action_count += 1
        action_type = action_dict.get("action_type", "wait")

        # --- Determine slider percentage ---
        if cycle <= 5:
            # Greenbiscuit opening: use specific slider values
            slider_pct = OPENING.get(cycle, 32)
        elif action_type == "defend":
            slider_pct = DEFEND_SLIDER_PCT
        elif action_type == "expand":
            slider_pct = EXPAND_PCT
        else:
            slider_pct = EXPAND_PCT

        result = {
            "executed": True,
            "action_type": action_type,
            "pixel_x": None,
            "pixel_y": None,
            "slider_pct": slider_pct,
            "safety_stop": False,
        }

        # --- Check safety stop: large territory drop ---
        state_before = await get_state(self.page)
        curr_territory = state_before["percentage"] / 100.0
        if self.prev_territory > 0:
            territory_drop = self.prev_territory - curr_territory
            if territory_drop > SAFETY_DROP_THRESHOLD:
                # Safety stop! Pause for 2 seconds
                self.safety_stops += 1
                result["safety_stop"] = True
                print(f"    ⚠️ SAFETY STOP #{self.safety_stops}: "
                      f"Territory dropped {territory_drop:.1%}! "
                      f"Pausing {SAFETY_PAUSE_MS}ms...")
                await self.page.wait_for_timeout(SAFETY_PAUSE_MS)

        # --- Execute the action ---
        if action_type == "click":
            await set_slider(self.page, slider_pct)
            pixel_x, pixel_y = self._translate_coordinates(
                action_dict.get("screen_x", 0.5),
                action_dict.get("screen_y", 0.5)
            )
            await self.page.mouse.click(pixel_x, pixel_y)
            result["pixel_x"] = pixel_x
            result["pixel_y"] = pixel_y

        elif action_type == "expand":
            await set_slider(self.page, slider_pct)
            # Click a random point centered around our base
            expand_x = random.randint(GAME_AREA_X_CENTER - 200, GAME_AREA_X_CENTER + 200)
            expand_y = random.randint(GAME_AREA_Y_CENTER - 200, GAME_AREA_Y_CENTER + 200)
            await self.page.mouse.click(expand_x, expand_y)
            result["pixel_x"] = expand_x
            result["pixel_y"] = expand_y

        elif action_type == "defend":
            await set_slider(self.page, DEFEND_SLIDER_PCT)
            # No click for defense — just set low slider and wait
            result["executed"] = True

        elif action_type == "wait":
            # Do nothing — just let the game tick
            await self.page.wait_for_timeout(DEFAULT_STEP_DELAY_MS)
            result["executed"] = True

        # Update territory tracking
        self.prev_territory = curr_territory

        return result

    async def verify_action(self, before_state: dict, after_state: dict) -> bool:
        """
        Verify that an action had a visible effect on the game state.
        
        Compares the game state before and after the action to determine
        if anything changed (territory, player count, etc.).
        
        Args:
            before_state: Game state dict from get_state() before action
            after_state:  Game state dict from get_state() after action
            
        Returns:
            True if the action had a visible effect, False otherwise
        """
        # Check if territory changed
        before_pct = before_state.get("percentage", 0)
        after_pct = after_state.get("percentage", 0)

        if abs(after_pct - before_pct) > 0.01:
            return True  # Territory changed measurably

        # Check if game state changed
        if before_state.get("game_over") != after_state.get("game_over"):
            return True

        # Check if time advanced (game is still running)
        if before_state.get("time") != after_state.get("time"):
            return True  # Time progressed, game is responsive

        return False

    def _translate_coordinates(self, norm_x: float, norm_y: float) -> tuple:
        """
        Translate normalized (0-1) coordinates to pixel coordinates
        centered around the base at (640, 450).
        
        Args:
            norm_x: Normalized X coordinate (0.0 to 1.0)
            norm_y: Normalized Y coordinate (0.0 to 1.0)
            
        Returns:
            Tuple of (pixel_x, pixel_y) in the browser viewport
        """
        # Clamp to valid range
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        # Map normalized [0,1] to centered pixel coordinates
        pixel_x = int(GAME_AREA_X_CENTER + (norm_x - 0.5) * 2 * GAME_AREA_X_RADIUS)
        pixel_y = int(GAME_AREA_Y_CENTER + (norm_y - 0.5) * 2 * GAME_AREA_Y_RADIUS)

        return pixel_x, pixel_y

    def get_stats(self) -> dict:
        """
        Get action controller statistics.
        
        Returns:
            Dict with action_count, safety_stops, etc.
        """
        return {
            "total_actions": self.action_count,
            "safety_stops": self.safety_stops,
            "last_territory": self.prev_territory,
        }


# ==============================================
# DEMO
# ==============================================

def demo():
    """Quick demo showing coordinate translation (no browser needed)."""
    print("\n🎮 Action Controller Demo")
    print("=" * 50)

    # Show coordinate translations
    print("\n📐 Coordinate Translation Examples:")
    print(f"   Viewport: {DEFAULT_VIEWPORT_WIDTH}x{DEFAULT_VIEWPORT_HEIGHT}")
    print(f"   Game area: ({GAME_AREA_X_START},{GAME_AREA_Y_START}) → "
          f"({GAME_AREA_X_END},{GAME_AREA_Y_END})")
    print()

    # Create controller without a real page (just for translation demo)
    class FakePage:
        pass

    controller = ActionController.__new__(ActionController)
    controller.viewport_width = DEFAULT_VIEWPORT_WIDTH
    controller.viewport_height = DEFAULT_VIEWPORT_HEIGHT

    test_coords = [
        (0.0, 0.0, "Top-left corner"),
        (0.5, 0.5, "Center"),
        (1.0, 1.0, "Bottom-right corner"),
        (0.25, 0.75, "Lower-left quadrant"),
    ]

    for nx, ny, label in test_coords:
        px, py = controller._translate_coordinates(nx, ny)
        print(f"   ({nx:.2f}, {ny:.2f}) → ({px}, {py})  [{label}]")

    print("\n🎮 Greenbiscuit Opening Slider Values:")
    for cycle, pct in OPENING.items():
        print(f"   Cycle {cycle}: {pct}%")

    print(f"\n   After cycle 5: {EXPAND_PCT}% (expansion mode)")
    print(f"   Defence mode:  {DEFEND_SLIDER_PCT}%")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
