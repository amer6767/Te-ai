"""
=============================================================================
game_environment.py — Territorial.io Playwright Game Environment
=============================================================================

Wraps the existing Playwright-based game loop into a clean class that can
be used interchangeably with brain.py's FakeGameEnvironment.

TO USE REAL GAME: replace FakeGameEnvironment with TerritorialEnvironment
in trainer.py

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

from playwright.async_api import async_playwright, Page, Browser
from PIL import Image
import numpy as np
import asyncio
import io
import random
import time

# ==============================================
# GAME CONSTANTS — Greenbiscuit opening strategy
# ==============================================

OPENING = {1: 30, 2: 35, 3: 32, 4: 30, 5: 46}
ATTACK_DENSITY_THRESHOLD = 50
EXPAND_PCT = 32
RED_INTEREST = 140

# ==============================================
# STANDALONE ASYNC FUNCTIONS
# ==============================================
# These can be called independently outside the class

async def get_state(page: Page) -> dict:
    """
    Read the current game state from the browser page.
    Parses territory percentage, time, game over status, and win status.
    """
    return await page.evaluate("""() => {
        try {
            const txt = document.body.innerText;
            const pctMatch = txt.match(/([\d.]+)%/);
            const timeMatch = txt.match(/(\d+:\d+)/);
            return {
                percentage: pctMatch ? parseFloat(pctMatch[1]) : 0,
                time: timeMatch ? timeMatch[1] : '0:00',
                game_over: txt.includes('Game Over') || txt.includes('You lost') || txt.includes('You win'),
                won: txt.includes('You win') || txt.includes('Winner')
            };
        } catch(e) {
            return { percentage: 0, time: '0:00', game_over: false, won: false };
        }
    }""")


async def set_slider(page: Page, pct: int):
    """
    Set the game's attack slider to the given percentage (0–100).
    Dispatches both 'input' and 'change' events for the browser to register.
    """
    await page.evaluate(f"""() => {{
        const inputs = document.querySelectorAll('input[type=range]');
        inputs.forEach(i => {{
            i.value = {pct};
            i.dispatchEvent(new Event('input', {{bubbles: true}}));
            i.dispatchEvent(new Event('change', {{bubbles: true}}));
        }});
    }}""")


async def ai_decide(page: Page, cycle: int, step: int) -> dict:
    """
    Legacy decision function from the original Playwright code.
    Makes a simple heuristic-based decision using the greenbiscuit opening
    for the first 5 cycles, then switches to expansion mode.
    
    Returns the game state dict after making the action.
    """
    state = await get_state(page)
    pct = state['percentage']

    if cycle <= 5:
        target_slider = OPENING.get(cycle, 32)
        await set_slider(page, target_slider)
        action_x = random.randint(270, 520)
        action_y = random.randint(220, 420)
        await page.mouse.click(action_x, action_y)
    else:
        await set_slider(page, EXPAND_PCT)
        action_x = random.randint(250, 550)
        action_y = random.randint(200, 500)
        await page.mouse.click(action_x, action_y)

    return state


# ==============================================
# TERRITORIAL ENVIRONMENT CLASS
# ==============================================

class TerritorialEnvironment:
    """
    Clean interface to the Territorial.io browser game.
    
    Provides reset/step/get_screenshot/close methods with the same
    info dict format as FakeGameEnvironment in brain.py, so trainer.py
    works with either environment without changes.
    
    Usage:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            env = TerritorialEnvironment(browser)
            screenshot = await env.reset()
            # ... game loop ...
            await env.close()
    """

    def __init__(self, browser: Browser):
        """
        Initialize the environment with a Playwright browser instance.
        
        Args:
            browser: Playwright Browser from chromium.launch()
        """
        self.browser = browser
        self.page = None
        self.step_count = 0
        self.cycle = 0
        self.max_steps = 200
        self.prev_territory = 0.0
        self.viewport_width = 1280
        self.viewport_height = 900
        self.ticks_per_cycle = 10

    async def reset(self) -> Image.Image:
        """
        Navigate to Territorial.io, set up a 30-player custom game,
        spawn at position (300, 280), and return the first screenshot.

        Uses selector-based clicking for UI buttons and a verification
        loop to confirm the player actually spawned before proceeding.

        Returns:
            PIL Image of the initial game state.
        """
        # Maximum number of full-page resets before giving up
        MAX_HARD_RESETS = 3

        for reset_attempt in range(MAX_HARD_RESETS):
            # --- Create a new browser page ---
            if self.page:
                await self.page.close()

            self.page = await self.browser.new_page(
                viewport={"width": self.viewport_width, "height": self.viewport_height}
            )

            # --- Navigate to the game ---
            await self.page.goto("https://territorial.io", timeout=30000)
            await self.page.wait_for_timeout(6000)

            # --- Select custom scenario (selector-based) ---
            try:
                custom_btn = self.page.get_by_text("Custom", exact=False)
                await custom_btn.wait_for(state="visible", timeout=10000)
                await custom_btn.click()
                await self.page.wait_for_timeout(2000)
            except Exception as e:
                print(f"   ⚠️ Could not find 'Custom' button: {e}")
                continue  # Hard reset

            # --- Set 30 players ---
            # Click the player count text box (roughly X: 300, Y: 280)
            await self.page.mouse.click(300, 280)
            await self.page.wait_for_timeout(500)
            
            # Delete existing number (e.g. 512)
            for _ in range(4):
                await self.page.keyboard.press("Backspace")
                
            # Type 30
            await self.page.keyboard.type("30")
            await self.page.wait_for_timeout(500)

            # --- Start the game (selector-based) ---
            try:
                start_btn = self.page.get_by_text("Play", exact=False)
                await start_btn.wait_for(state="visible", timeout=6000)
                await start_btn.click()
            except Exception as e:
                print(f"   ⚠️ Could not find 'Play' button, falling back to coordinates: {e}")
                # Fallback to roughly X: 960, Y: 860
                await self.page.mouse.click(960, 860)
            
            await self.page.wait_for_timeout(4000)

            # --- Spawn verification loop ---
            spawn_success = await self._verify_spawn()
            if spawn_success:
                break  # Spawn confirmed — proceed

            # Spawn failed after timeout — save debug screenshot & hard reset
            print(f"   🔄 Hard reset {reset_attempt + 1}/{MAX_HARD_RESETS}")

        else:
            # All hard resets exhausted
            print("   ❌ Failed to spawn after all reset attempts")

        # --- Reset tracking ---
        self.step_count = 0
        self.cycle = 0
        self.prev_territory = 0.0

        # Return the initial screenshot as PIL Image
        screenshot = await self.get_screenshot()
        
        # Save to show the first frame
        try:
            screenshot.save("live_view.png")
        except Exception:
            pass
            
        return screenshot

    async def _verify_spawn(self, timeout_seconds: float = 5.0) -> bool:
        """
        Attempt to spawn and verify that territory > 0%.

        Retries the double-click with slight coordinate jitter until
        spawn is confirmed or the timeout is reached.

        Args:
            timeout_seconds: Maximum time to keep retrying.

        Returns:
            True if spawn succeeded (territory > 0%), False otherwise.
        """
        SPAWN_X, SPAWN_Y = 300, 280
        JITTER = 15  # pixels of random jitter on retry
        start_time = time.time()
        attempt = 0

        while (time.time() - start_time) < timeout_seconds:
            # Double-click to spawn (add jitter after first attempt)
            jx = SPAWN_X + (random.randint(-JITTER, JITTER) if attempt > 0 else 0)
            jy = SPAWN_Y + (random.randint(-JITTER, JITTER) if attempt > 0 else 0)
            await self.page.mouse.dblclick(jx, jy)
            await self.page.wait_for_timeout(500)

            # Check if we actually spawned
            state = await get_state(self.page)
            if state["percentage"] > 0:
                print(f"   ✅ Spawn confirmed ({state['percentage']}%) "
                      f"on attempt {attempt + 1}")
                # Zoom out so the CNN model can see more of the map
                await self.page.mouse.wheel(0, 2000)
                await self.page.wait_for_timeout(500)
                return True

            attempt += 1

        # Timeout — save debug screenshot
        try:
            await self.page.screenshot(path="debug_spawn_failure.png")
            print("   📸 Debug screenshot saved: debug_spawn_failure.png")
        except Exception as e:
            print(f"   ⚠️ Could not save debug screenshot: {e}")

        print(f"   ❌ Spawn verification timed out after {timeout_seconds}s "
              f"({attempt} attempts)")
        return False

    async def step(self, action_dict: dict) -> tuple:
        """
        Execute an action from decision.py in the game.
        
        Args:
            action_dict: Action dictionary from GameAgent.select_action(), with keys:
                action_type: "click", "expand", "defend", "wait"
                screen_x: normalized X (0-1) for click actions
                screen_y: normalized Y (0-1) for click actions
                grid_row, grid_col: grid position for click actions
        
        Returns:
            Tuple of (screenshot, reward, done, info) where:
                screenshot: PIL Image of current game state
                reward:     float reward based on territory change
                done:       bool indicating game over
                info:       dict with {territory, step, won, game_over, cycle, num_players}
                            — same format as FakeGameEnvironment
        """
        self.step_count += 1
        self.cycle = (self.step_count // self.ticks_per_cycle) + 1

        action_type = action_dict.get("action_type", "wait")

        # --- Apply Greenbiscuit opening override for cycles 1-5 ---
        if self.cycle <= 5:
            slider_pct = OPENING.get(self.cycle, 32)
        else:
            slider_pct = EXPAND_PCT

        # --- Execute the action based on type ---
        if action_type == "click":
            # Set slider according to current strategy
            await set_slider(self.page, slider_pct)
            # Translate normalized coordinates to pixel coordinates
            screen_x = action_dict.get("screen_x", 0.5)
            screen_y = action_dict.get("screen_y", 0.5)
            # Map normalized [0,1] to game area pixels (centered around base)
            pixel_x = int(640 + (screen_x - 0.5) * 800)
            pixel_y = int(450 + (screen_y - 0.5) * 600)
            await self.page.mouse.click(pixel_x, pixel_y)

        elif action_type == "expand":
            await set_slider(self.page, EXPAND_PCT)
            # Click a random area to expand into (centered around base)
            expand_x = random.randint(640 - 200, 640 + 200)
            expand_y = random.randint(450 - 200, 450 + 200)
            await self.page.mouse.click(expand_x, expand_y)

        elif action_type == "defend":
            # Low slider value, stay defensive
            await set_slider(self.page, 10)
            # No click — just wait

        elif action_type == "wait":
            # Simply wait, do nothing
            pass

        # --- Wait for game to process the action ---
        await self.page.wait_for_timeout(800)

        # --- Read the new game state ---
        state = await get_state(self.page)
        curr_territory = state["percentage"] / 100.0  # Convert from % to decimal

        # --- Calculate reward from territory change ---
        territory_change = curr_territory - self.prev_territory
        reward = territory_change * 100.0  # Scale for meaningful signal
        self.prev_territory = curr_territory

        # --- Check if game is over ---
        done = state["game_over"] or self.step_count >= self.max_steps

        # --- Get screenshot ---
        screenshot = await self.get_screenshot()

        # Save a frame every 6 moves (~5 seconds)
        if self.step_count % 6 == 0:
            try:
                screenshot.save("live_view.png")
            except Exception:
                pass

        # --- Build info dict — IDENTICAL format to FakeGameEnvironment ---
        info = {
            "territory": curr_territory,
            "step": self.step_count,
            "won": state["won"],
            "game_over": state["game_over"],
            "cycle": self.cycle,
            "num_players": 30,  # We set 30 at start; could estimate from screen
        }

        return screenshot, reward, done, info

    async def get_screenshot(self) -> Image.Image:
        """
        Capture the current browser page as a PIL Image.
        
        Returns:
            PIL Image in RGB format.
        """
        screenshot_bytes = await self.page.screenshot()
        image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        return image

    async def close(self):
        """Close the browser page."""
        if self.page:
            await self.page.close()
            self.page = None


# ==============================================
# DEMO / TESTING
# ==============================================

async def demo():
    """
    Quick demo: launch browser, play one abbreviated game.
    Run with: python -c "import asyncio; from game_environment import demo; asyncio.run(demo())"
    """
    print("\n🎮 Game Environment Demo")
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

            for step in range(10):
                # Simple random action for demo
                action = {
                    "action_type": "click",
                    "action_index": random.randint(0, 255),
                    "screen_x": random.uniform(0.2, 0.8),
                    "screen_y": random.uniform(0.2, 0.8),
                    "grid_row": random.randint(0, 15),
                    "grid_col": random.randint(0, 15),
                    "was_random": True,
                    "confidence": 0.0,
                }
                screenshot, reward, done, info = await env.step(action)
                print(f"  Step {step+1}: territory={info['territory']:.2%}, "
                      f"reward={reward:.2f}, done={done}")
                if done:
                    break

        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            await env.close()
            await browser.close()

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo())
