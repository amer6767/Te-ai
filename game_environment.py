"""
=============================================================================
game_environment.py — Territorial.io Playwright Game Environment (v2)
=============================================================================

STATE-AWARE AI — reads Troops, Interest, Time, Cycle/Tick from the DOM.
Executes a hardcoded Greenbiscuit opening for Cycles 1-5, then hands
control to the neural network with smart pixel-based targeting.

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

from playwright.async_api import async_playwright, Page, Browser
from PIL import Image
from screen_capture import ScreenCapture
import numpy as np
import asyncio
import io
import random
import time
import colorsys
import math

# ==============================================
# GAME CONSTANTS
# ==============================================

# --- Greenbiscuit Opening Table (from tips.txt) ---
# Each entry: (cycle, tick_to_attack, slider_pct)
GREENBISCUIT_OPENING = [
    {"cycle": 1, "tick": 7, "slider": 30},
    {"cycle": 2, "tick": 7, "slider": 35},
    {"cycle": 3, "tick": 7, "slider": 32},
    {"cycle": 4, "tick": 7, "slider": 30},
    {"cycle": 5, "tick": 6, "slider": 46},
]

# --- Post-Opening Constants ---
EXPAND_PCT = 32              # Default expansion slider
INFINITE_EXPAND_TICK = 3     # Attack at tick 3 each cycle after opening
INFINITE_EXPAND_PCT = 32     # Slider for infinite expansion
RED_INTEREST_DENSITY = 140   # Density threshold for red interest danger

# --- Viewport ---
VIEWPORT_W = 1280
VIEWPORT_H = 900

# --- Player color detection (HSV ranges for "our" blue territory) ---
PLAYER_HUE_MIN = 200
PLAYER_HUE_MAX = 260
PLAYER_SAT_MIN = 30

# --- Neutral land detection (gray/white = low saturation) ---
NEUTRAL_SAT_MAX = 15
NEUTRAL_VAL_MIN = 40
NEUTRAL_VAL_MAX = 240


# ==============================================
# ENHANCED GAME STATE READER
# ==============================================

async def get_state(page: Page) -> dict:
    """
    Read the FULL game state from the browser DOM.
    
    Returns dict with:
        percentage:    float — territory %
        time:          str   — game time "M:SS"
        troops:        int   — current troop balance
        interest:      float — interest rate %
        red_interest:  bool  — whether we are in red interest
        game_over:     bool
        won:           bool
        density:       float — troops / land proxy
    """
    return await page.evaluate(r"""() => {
        try {
            const txt = document.body.innerText;
            
            // --- Territory % ---
            const pctMatch = txt.match(/([\d.]+)%/);
            const percentage = pctMatch ? parseFloat(pctMatch[1]) : 0;
            
            // --- Game Time ---
            const timeMatch = txt.match(/(\d+:\d{2})/);
            const gameTime = timeMatch ? timeMatch[1] : '0:00';
            
            // --- Troops (Balance) ---
            // The troop count is typically a large number on screen
            const troopMatches = txt.match(/\b(\d[\d,]{2,})\b/g);
            let troops = 0;
            if (troopMatches) {
                // Pick the largest number found (likely the troop balance)
                for (const m of troopMatches) {
                    const val = parseInt(m.replace(/,/g, ''));
                    if (val > troops && val < 10000000) troops = val;
                }
            }
            
            // --- Interest Rate ---
            // Look for patterns like "5.00%" or similar near "interest"
            const interestMatch = txt.match(/([\d.]+)%/g);
            let interest = 5.0;  // default
            if (interestMatch && interestMatch.length >= 2) {
                // Second percentage is usually interest (first is territory)
                interest = parseFloat(interestMatch[1]);
            }
            
            // --- Red Interest Detection ---
            // Check if any element with the troop count has red text
            let redInterest = false;
            try {
                const allElements = document.querySelectorAll('*');
                for (const el of allElements) {
                    const style = window.getComputedStyle(el);
                    const color = style.color;
                    if (color && el.innerText && el.innerText.match(/\d{3,}/)) {
                        // Check for red-ish color
                        const rgb = color.match(/(\d+)/g);
                        if (rgb && parseInt(rgb[0]) > 200 && parseInt(rgb[1]) < 100 && parseInt(rgb[2]) < 100) {
                            redInterest = true;
                            break;
                        }
                    }
                }
            } catch(e) {}
            
            // --- Game Over ---
            const gameOver = txt.includes('Game Over') || txt.includes('You lost') || txt.includes('You win');
            const won = txt.includes('You win') || txt.includes('Winner');
            
            return {
                percentage: percentage,
                time: gameTime,
                troops: troops,
                interest: interest,
                red_interest: redInterest,
                game_over: gameOver,
                won: won,
                density: 0  // Will be calculated in Python
            };
        } catch(e) {
            return { 
                percentage: 0, time: '0:00', troops: 0, interest: 5.0,
                red_interest: false, game_over: false, won: false, density: 0 
            };
        }
    }""")


async def set_slider(page: Page, pct: int):
    """Set the game's attack slider to the given percentage (0–100)."""
    await page.evaluate(f"""() => {{
        const inputs = document.querySelectorAll('input[type=range]');
        inputs.forEach(i => {{
            i.value = {pct};
            i.dispatchEvent(new Event('input', {{bubbles: true}}));
            i.dispatchEvent(new Event('change', {{bubbles: true}}));
        }});
    }}""")


# ==============================================
# SMART VISION TARGETING (The "Aimbot")
# ==============================================

class SmartTargeting:
    """
    Analyzes game screenshots to find valid attack targets.

    Supports two targeting modes:
      - get_best_target():      Raycast from center, find nearest border (any direction)
      - get_target_in_zone():   Scan inside a specific 3×3 grid zone for attack targets
    """
    
    @staticmethod
    def get_best_target(image: Image.Image, preferred_type: str = "neutral") -> tuple:
        """
        Scan outward from the center to find the edge of our territory.
        
        Args:
            image: PIL Image of the game (1280x900)
            preferred_type: Ignored for raycast (treats all borders equally)
        
        Returns:
            (pixel_x, pixel_y) — the best spot to click, just past the border
        """
        pixels = np.array(image)
        cx, cy = VIEWPORT_W // 2, VIEWPORT_H // 2
        
        # 1. Memorize our exact color at the dead center of the screen
        player_color = pixels[cy, cx].astype(int)
        
        # 2. Pick 8 random directions to look (0, 45, 90, 135... degrees)
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        random.shuffle(angles)
        
        # 3. Walk outward from the center like a radar
        for angle in angles:
            rad = math.radians(angle)
            for dist in range(5, 400, 5):
                x = int(cx + math.cos(rad) * dist)
                y = int(cy + math.sin(rad) * dist)
                
                if y < 50 or y > 850 or x < 10 or x > 1270:
                    break
                    
                current_color = pixels[y, x].astype(int)
                color_diff = sum((current_color - player_color) ** 2)
                
                if color_diff > 1000:
                    click_x = int(cx + math.cos(rad) * (dist + 15))
                    click_y = int(cy + math.sin(rad) * (dist + 15))
                    return (click_x, click_y)
        
        return (cx + random.randint(-30, 30), cy + random.randint(-30, 30))

    @staticmethod
    def get_target_in_zone(image: Image.Image, zone_id: int) -> tuple:
        """
        Find the best attack target INSIDE a specific zone (1-9).

        Calculates the pixel boundaries of the requested zone, then
        shoots rays from the center of the screen toward that zone to
        find the nearest border pixel within the zone's bounds.

        If no border is found inside the zone, falls back to clicking
        the center of the zone.

        Args:
            image:   PIL Image of the game (1280x900)
            zone_id: Zone number 1-9 from the LLM command

        Returns:
            (pixel_x, pixel_y) — the spot to click inside that zone
        """
        zone_id = max(1, min(9, zone_id))

        # Get zone boundaries from ScreenCapture
        x_start, y_start, x_end, y_end = ScreenCapture.get_zone_bounds(zone_id)
        zone_cx, zone_cy = ScreenCapture.get_zone_center(zone_id)

        pixels = np.array(image)
        img_h, img_w = pixels.shape[:2]
        screen_cx, screen_cy = VIEWPORT_W // 2, VIEWPORT_H // 2

        # Memorize our color at center
        player_color = pixels[screen_cy, screen_cx].astype(int)

        # Shoot a ray from our base (center) toward the zone center
        dx = zone_cx - screen_cx
        dy = zone_cy - screen_cy
        ray_length = math.sqrt(dx * dx + dy * dy)

        if ray_length < 1:
            # Zone 5 (center) — just do a best-target scan
            return SmartTargeting.get_best_target(image)

        # Normalize direction
        ndx = dx / ray_length
        ndy = dy / ray_length

        # Walk along the ray, looking for a border INSIDE the zone
        best_target = None
        for dist in range(5, int(ray_length) + 100, 5):
            x = int(screen_cx + ndx * dist)
            y = int(screen_cy + ndy * dist)

            # Bounds check
            if x < 0 or x >= img_w or y < 0 or y >= img_h:
                break

            # Is this pixel inside the target zone?
            if x_start <= x <= x_end and y_start <= y <= y_end:
                current_color = pixels[y, x].astype(int)
                color_diff = sum((current_color - player_color) ** 2)

                if color_diff > 1000:
                    # Border found inside the zone!
                    click_x = int(screen_cx + ndx * (dist + 15))
                    click_y = int(screen_cy + ndy * (dist + 15))
                    click_x = max(x_start, min(x_end, click_x))
                    click_y = max(y_start, min(y_end, click_y))
                    return (click_x, click_y)

        # No border found along primary ray — try scanning within the zone
        # Shoot rays from zone center outward in 8 directions
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            rad = math.radians(angle)
            for dist in range(5, 200, 5):
                x = int(zone_cx + math.cos(rad) * dist)
                y = int(zone_cy + math.sin(rad) * dist)

                # Stay inside zone bounds
                if x < x_start or x > x_end or y < y_start or y > y_end:
                    break
                if x < 0 or x >= img_w or y < 0 or y >= img_h:
                    break

                current_color = pixels[y, x].astype(int)
                color_diff = sum((current_color - player_color) ** 2)

                if color_diff > 1000:
                    click_x = int(zone_cx + math.cos(rad) * (dist + 10))
                    click_y = int(zone_cy + math.sin(rad) * (dist + 10))
                    click_x = max(x_start, min(x_end, click_x))
                    click_y = max(y_start, min(y_end, click_y))
                    return (click_x, click_y)

        # Absolute fallback: click the center of the zone
        return (zone_cx, zone_cy)


# ==============================================
# TERRITORIAL ENVIRONMENT CLASS (v2)
# ==============================================

class TerritorialEnvironment:
    """
    State-Aware interface to the Territorial.io browser game.
    
    Key improvements over v1:
      - Reads Troops, Interest, Time, Cycle/Tick from DOM
      - Executes hardcoded Greenbiscuit opening for Cycles 1-5
      - Uses pixel-based smart targeting (no more random clicks)
      - Detects Red Interest and forces defensive play
      - After opening, runs "infinite expansion" mode
    """

    def __init__(self, browser: Browser):
        self.browser = browser
        self.page = None
        self.step_count = 0
        self.max_steps = 200
        self.prev_territory = 0.0
        self.viewport_width = VIEWPORT_W
        self.viewport_height = VIEWPORT_H
        
        # --- State tracking ---
        self.game_start_time = None
        self.current_cycle = 0
        self.current_tick = 0
        self.opening_attacks_done = set()  # Track which opening steps we've fired
        self.opening_complete = False
        self.last_troops = 0
        self.last_interest = 5.0

    # ------------------------------------------
    # TIME / CYCLE / TICK HELPERS
    # ------------------------------------------

    @staticmethod
    def parse_game_time(time_str: str) -> int:
        """Convert 'M:SS' game time to total seconds."""
        try:
            parts = time_str.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError):
            pass
        return 0

    @staticmethod
    def get_cycle_and_tick(total_seconds: int) -> tuple:
        """
        Calculate the current Cycle and Tick from total game seconds.
        
        Each Cycle = 10 Ticks.
        Tick 1 = start of cycle, Tick 10 = end of cycle.
        
        Returns:
            (cycle, tick) — both 1-indexed
        """
        if total_seconds <= 0:
            return (1, 1)
        cycle = (total_seconds // 10) + 1
        tick = (total_seconds % 10) + 1
        return (cycle, tick)

    # ------------------------------------------
    # RESET
    # ------------------------------------------

    async def reset(self) -> Image.Image:
        """
        Navigate to Territorial.io, set up a custom game,
        spawn, and return the first screenshot.
        """
        MAX_HARD_RESETS = 3

        for reset_attempt in range(MAX_HARD_RESETS):
            if self.page:
                await self.page.close()

            self.page = await self.browser.new_page(
                viewport={"width": self.viewport_width, "height": self.viewport_height}
            )

            # --- Navigate ---
            await self.page.goto("https://territorial.io", timeout=30000)
            await self.page.wait_for_timeout(6000)

            # --- Select custom scenario ---
            try:
                custom_btn = self.page.get_by_text("Custom", exact=False)
                await custom_btn.wait_for(state="visible", timeout=10000)
                await custom_btn.click()
                await self.page.wait_for_timeout(2000)
            except Exception as e:
                print(f"   ⚠️ Could not find 'Custom' button: {e}")
                continue

            # --- Set 30 players ---
            await self.page.mouse.click(300, 280)
            await self.page.wait_for_timeout(500)
            for _ in range(4):
                await self.page.keyboard.press("Backspace")
            await self.page.keyboard.type("30")
            await self.page.wait_for_timeout(500)

            # --- Start the game ---
            try:
                start_btn = self.page.get_by_text("Play", exact=False)
                await start_btn.wait_for(state="visible", timeout=6000)
                await start_btn.click()
            except Exception as e:
                print(f"   ⚠️ Could not find 'Play', falling back to coordinates: {e}")
                await self.page.mouse.click(960, 860)
            
            await self.page.wait_for_timeout(4000)

            # --- Spawn verification ---
            spawn_success = await self._verify_spawn()
            if spawn_success:
                break

            print(f"   🔄 Hard reset {reset_attempt + 1}/{MAX_HARD_RESETS}")
        else:
            print("   ❌ Failed to spawn after all reset attempts")

        # --- Reset all state tracking ---
        self.step_count = 0
        self.prev_territory = 0.0
        self.opening_attacks_done = set()
        self.opening_complete = False
        self.last_troops = 0
        self.last_interest = 5.0
        
        # Record the game start time
        state = await get_state(self.page)
        self.game_start_time = time.time()
        total_sec = self.parse_game_time(state.get("time", "0:00"))
        self.current_cycle, self.current_tick = self.get_cycle_and_tick(total_sec)
        
        print(f"   🕐 Game started at {state.get('time', '0:00')} "
              f"(Cycle {self.current_cycle}, Tick {self.current_tick})")
        print(f"   💰 Troops: {state.get('troops', 0)} | "
              f"Interest: {state.get('interest', 0)}% | "
              f"Red: {state.get('red_interest', False)}")

        screenshot = await self.get_screenshot()
        try:
            screenshot.save("live_view.png")
        except Exception:
            pass
        return screenshot

    async def _verify_spawn(self, timeout_seconds: float = 5.0) -> bool:
        """Attempt to spawn and verify territory > 0%."""
        SPAWN_X, SPAWN_Y = 300, 280
        JITTER = 15
        start_time = time.time()
        attempt = 0

        while (time.time() - start_time) < timeout_seconds:
            jx = SPAWN_X + (random.randint(-JITTER, JITTER) if attempt > 0 else 0)
            jy = SPAWN_Y + (random.randint(-JITTER, JITTER) if attempt > 0 else 0)
            await self.page.mouse.dblclick(jx, jy)
            await self.page.wait_for_timeout(500)

            state = await get_state(self.page)
            if state["percentage"] > 0:
                print(f"   ✅ Spawn confirmed ({state['percentage']}%) "
                      f"on attempt {attempt + 1}")
                # Move mouse to center BEFORE scrolling
                await self.page.mouse.move(640, 450)
                await self.page.mouse.wheel(0, 5000) # Big scroll to zoom out
                await self.page.wait_for_timeout(500)
                return True

            attempt += 1

        try:
            await self.page.screenshot(path="debug_spawn_failure.png")
            print("   📸 Debug screenshot saved: debug_spawn_failure.png")
        except Exception as e:
            print(f"   ⚠️ Could not save debug screenshot: {e}")

        print(f"   ❌ Spawn timed out after {timeout_seconds}s ({attempt} attempts)")
        return False

    # ------------------------------------------
    # STEP — The core game loop tick
    # ------------------------------------------

    async def step(self, command: dict) -> tuple:
        """
        Execute one game step. The logic is:
        
        1. Read full game state (troops, time, cycle, tick, interest)
        2. If Cycles 1-5: execute Greenbiscuit opening at exact ticks
        3. If Red Interest: force defend (wait, recover troops)
        4. If Cycle 6+: "infinite expansion" at tick 3, else LLM decides
        5. Use smart pixel targeting for all clicks
        
        Args:
            command: dict from LLMCommander.decide() with keys:
                     "target_zone" (int, 1-9 or 0=wait) and "slider_pct" (int)
        
        Returns:
            (screenshot, reward, done, info)
        """
        self.step_count += 1

        # ============================================
        # 1. READ FULL GAME STATE
        # ============================================
        state = await get_state(self.page)
        total_sec = self.parse_game_time(state.get("time", "0:00"))
        self.current_cycle, self.current_tick = self.get_cycle_and_tick(total_sec)
        
        curr_territory = state["percentage"] / 100.0
        troops = state.get("troops", 0)
        interest = state.get("interest", 5.0)
        red_interest = state.get("red_interest", False)
        
        self.last_troops = troops
        self.last_interest = interest

        # ============================================
        # 2. DECIDE ACTION — Opening vs LLM Commander
        # ============================================
        action_taken = "wait"
        click_x, click_y = None, None
        slider_used = 0
        
        # --- SAFETY: Red Interest → force defend ---
        if red_interest and self.opening_complete:
            await set_slider(self.page, 10)
            action_taken = "defend_red_interest"
        
        # --- OPENING PHASE (Cycles 1-5) ---
        elif self.current_cycle <= 5 and not self.opening_complete:
            action_taken, click_x, click_y, slider_used = await self._execute_opening()
            
            # Mark opening complete after cycle 5
            if self.current_cycle >= 5 and len(self.opening_attacks_done) >= 5:
                self.opening_complete = True
                print("   🎯 Opening complete! Switching to LLM Commander + infinite expansion")
        
        # --- POST-OPENING: Infinite Expansion + LLM Commander ---
        else:
            action_taken, click_x, click_y, slider_used = await self._execute_post_opening(
                command
            )

        # ============================================
        # 3. WAIT FOR GAME TO PROCESS
        # ============================================
        await self.page.wait_for_timeout(800)

        # ============================================
        # 4. READ NEW STATE & CALCULATE REWARD
        # ============================================
        new_state = await get_state(self.page)
        new_territory = new_state["percentage"] / 100.0
        
        territory_change = new_territory - self.prev_territory
        
        # Enhanced reward signal
        reward = territory_change * 100.0  # Base territory reward
        
        # Bonus for maintaining positive interest
        if not new_state.get("red_interest", False) and interest > 3.0:
            reward += 0.5  # Small bonus for healthy economy
        
        # Penalty for red interest
        if new_state.get("red_interest", False):
            reward -= 2.0
        
        # Bonus for troop growth
        new_troops = new_state.get("troops", 0)
        if new_troops > troops and troops > 0:
            reward += 0.3  # Troops are growing
        
        self.prev_territory = new_territory

        # ============================================
        # 5. CHECK GAME OVER
        # ============================================
        done = new_state["game_over"] or self.step_count >= self.max_steps

        # ============================================
        # 6. SCREENSHOT
        # ============================================
        screenshot = await self.get_screenshot()
        
        if self.step_count % 3 == 0:
            try:
                screenshot.save("live_view.png")
            except Exception:
                pass

        # ============================================
        # 7. BUILD INFO DICT
        # ============================================
        info = {
            "territory": new_territory,
            "step": self.step_count,
            "won": new_state["won"],
            "game_over": new_state["game_over"],
            "cycle": self.current_cycle,
            "tick": self.current_tick,
            "troops": new_troops,
            "interest": new_state.get("interest", 0),
            "red_interest": new_state.get("red_interest", False),
            "action_taken": action_taken,
            "opening_complete": self.opening_complete,
            "num_players": 30,
        }
        
        # Log key moments
        if self.step_count % 10 == 0:
            print(f"   📊 Step {self.step_count} | C{self.current_cycle}:T{self.current_tick} | "
                  f"Territory: {new_territory:.2%} | Troops: {new_troops} | "
                  f"Interest: {new_state.get('interest', 0)}% | "
                  f"Action: {action_taken}")

        return screenshot, reward, done, info

    # ------------------------------------------
    # OPENING LOGIC (Greenbiscuit)
    # ------------------------------------------

    async def _execute_opening(self) -> tuple:
        """
        Execute the Greenbiscuit opening.
        
        Waits for the exact tick, then sets slider and clicks nearest
        neutral land. Does NOT attack at other ticks (saves troops).
        
        Returns:
            (action_name, click_x, click_y, slider_pct)
        """
        for step_info in GREENBISCUIT_OPENING:
            cycle = step_info["cycle"]
            tick = step_info["tick"]
            slider = step_info["slider"]
            
            # Check if this is the right cycle AND tick, and we haven't done it yet
            if (self.current_cycle == cycle and 
                self.current_tick >= tick and
                cycle not in self.opening_attacks_done):
                
                # Set slider
                await set_slider(self.page, slider)
                await self.page.wait_for_timeout(100)
                
                # Use smart targeting to find nearest neutral land
                screenshot = await self.get_screenshot()
                target_x, target_y = SmartTargeting.get_best_target(
                    screenshot, preferred_type="neutral"
                )
                
                # Click!
                await self.page.mouse.click(target_x, target_y)
                self.opening_attacks_done.add(cycle)
                
                print(f"   🎯 OPENING: Cycle {cycle}, Tick {tick} → "
                      f"Slider {slider}% → Click ({target_x}, {target_y})")
                
                return (f"opening_c{cycle}", target_x, target_y, slider)
        
        # Not the right tick yet — WAIT (save troops!)
        return ("wait_for_tick", None, None, 0)

    # ------------------------------------------
    # POST-OPENING LOGIC (Infinite Expansion + LLM)
    # ------------------------------------------

    async def _execute_post_opening(self, command: dict) -> tuple:
        """
        After the opening is complete (Cycle 6+):
        
        1. At Tick ~3 each cycle: "infinite expansion" — 
           auto-attack neutral land with 30-35% (saves LLM calls)
        2. Other ticks: execute the LLM Commander's zone + slider
        3. Uses SmartTargeting.get_target_in_zone() for precise clicks
        
        Args:
            command: dict from LLMCommander.decide() with:
                     "target_zone" (int, 1-9 or 0=wait)
                     "slider_pct" (int, 0-100)
        
        Returns:
            (action_name, click_x, click_y, slider_pct)
        """
        target_zone = command.get("target_zone", 0)
        slider_pct = command.get("slider_pct", 0)
        
        # --- Infinite Expansion at Tick 2-4 (auto, no LLM needed) ---
        if self.current_tick in (2, 3, 4):
            expand_slider = random.choice([30, 32, 35])
            await set_slider(self.page, expand_slider)
            await self.page.wait_for_timeout(100)
            
            screenshot = await self.get_screenshot()
            target_x, target_y = SmartTargeting.get_best_target(
                screenshot, preferred_type="neutral"
            )
            
            await self.page.mouse.click(target_x, target_y)
            return ("infinite_expand", target_x, target_y, expand_slider)
        
        # --- LLM Commander decides ---
        if target_zone == 0:
            return ("llm_wait", None, None, 0)
        
        # Attack in the zone the LLM chose
        await set_slider(self.page, slider_pct)
        await self.page.wait_for_timeout(100)
        
        screenshot = await self.get_screenshot()
        target_x, target_y = SmartTargeting.get_target_in_zone(
            screenshot, target_zone
        )
        
        await self.page.mouse.click(target_x, target_y)
        return (f"llm_attack_zone{target_zone}", target_x, target_y, slider_pct)

    # ------------------------------------------
    # SCREENSHOT
    # ------------------------------------------

    async def get_screenshot(self) -> Image.Image:
        """Capture the current browser page as a PIL Image."""
        screenshot_bytes = await self.page.screenshot()
        image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        return image

    async def close(self):
        """Close the browser page."""
        if self.page:
            await self.page.close()
            self.page = None


# ==============================================
# LEGACY FUNCTION (kept for backwards compat)
# ==============================================

async def ai_decide(page: Page, cycle: int, step: int) -> dict:
    """Legacy decision function. Use TerritorialEnvironment.step() instead."""
    state = await get_state(page)
    pct = state['percentage']

    if cycle <= 5:
        target_slider = GREENBISCUIT_OPENING[min(cycle - 1, 4)]["slider"]
        await set_slider(page, target_slider)
        action_x = random.randint(440, 840)
        action_y = random.randint(250, 650)
        await page.mouse.click(action_x, action_y)
    else:
        await set_slider(page, EXPAND_PCT)
        action_x = random.randint(440, 840)
        action_y = random.randint(250, 650)
        await page.mouse.click(action_x, action_y)

    return state


# ==============================================
# DEMO / TESTING
# ==============================================

async def demo():
    """Quick demo: launch browser, play one abbreviated game."""
    print("\n🎮 Game Environment v2 Demo")
    print("=" * 50)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )

        env = TerritorialEnvironment(browser)
        print("📡 Connecting to Territorial.io...")

        try:
            screenshot = await env.reset()
            print(f"✅ Game started! Screenshot size: {screenshot.size}")

            for step in range(20):
                command = {
                    "target_zone": random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9]),
                    "slider_pct": random.randint(10, 60),
                }
                screenshot, reward, done, info = await env.step(command)
                print(f"  Step {step+1}: C{info['cycle']}:T{info['tick']} | "
                      f"territory={info['territory']:.2%} | troops={info['troops']} | "
                      f"action={info['action_taken']} | reward={reward:.2f}")
                if done:
                    break

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await env.close()
            await browser.close()

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo())
