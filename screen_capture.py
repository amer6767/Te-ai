"""
=============================================================================
screen_capture.py — Browser Screenshot Processing for Territorial.io AI
=============================================================================

Processes raw browser screenshots into usable data for Nemotron LLM:
  - Text-based Radar Report (8-direction raycast scan)
  - Territory percentage estimates from color analysis
  - Player count estimates from distinct colors
  - Cropped game area with browser UI removed
  - Territory trend tracking for momentum analysis

The radar report replaces CNN tensors — Nemotron reads a plain-English
spatial description instead of processing pixel arrays.

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

from PIL import Image
import numpy as np
import math
import io
import colorsys
from collections import Counter

# Try to import Playwright page type
try:
    from playwright.async_api import Page
except ImportError:
    Page = None  # Running without Playwright installed


# ==============================================
# CONFIGURATION
# ==============================================

# Game area boundaries (approximate, browser UI removed)
GAME_AREA_LEFT = 0
GAME_AREA_TOP = 50       # Remove top browser/game bar
GAME_AREA_RIGHT = 1280
GAME_AREA_BOTTOM = 850   # Remove bottom UI bar

# Player color: Territorial.io uses blue for the player's territory
# These are HSV ranges for detecting "our" territory
PLAYER_HUE_MIN = 200     # Blue hue range in degrees
PLAYER_HUE_MAX = 260
PLAYER_SAT_MIN = 40      # Minimum saturation (percentage)
PLAYER_VAL_MIN = 40      # Minimum value/brightness (percentage)

# Neutral territory (unclaimed) is typically gray
NEUTRAL_SAT_MAX = 15     # Very low saturation = gray

# Minimum distinct pixels to count as a player color
MIN_PLAYER_PIXELS = 100

# ---- Grid Configuration (3x3 = 9 Zones) ----
# The game area (1280x800 after cropping top/bottom UI) is split into a 3x3 grid.
# Each zone is approximately 427x267 pixels.
GRID_COLS = 3
GRID_ROWS = 3

# Zone names for human-readable output
ZONE_NAMES = {
    1: "Top-Left",    2: "Top-Center",    3: "Top-Right",
    4: "Center-Left", 5: "Center",        6: "Center-Right",
    7: "Bot-Left",    8: "Bot-Center",    9: "Bot-Right",
}


# ==============================================
# SCREEN CAPTURE CLASS
# ==============================================

class ScreenCapture:
    """
    Processes Playwright browser screenshots for the AI system.

    Provides:
    - Grid Report: 3×3 zone spatial analysis for the LLM
    - Territory percentage estimation from blue-pixel analysis
    - Player count estimation from distinct hue clusters
    - Territory trend tracking across frames

    Usage:
        capture = ScreenCapture(page)
        frame = await capture.get_processed_frame()
        grid   = frame["grid_text"]        # LLM reads this
        terr   = frame["territory_pct"]    # Our territory percentage
        players = frame["num_players"]     # Estimated player count
    """

    def __init__(self, page):
        """
        Initialize with a Playwright page instance.

        Args:
            page: Playwright Page object for taking screenshots
        """
        self.page = page
        self._frame_count = 0
        self._last_territory = 0.0
        self._territory_history = []

    async def get_processed_frame(self) -> dict:
        """
        Capture and process the current browser screenshot.

        Returns:
            Dictionary with:
                raw_image:       PIL Image — full browser screenshot
                territory_pct:   float — estimated player territory % (0.0–1.0)
                num_players:     int — estimated number of distinct players
                game_area:       PIL Image — cropped to game map only
                grid_text:       str — 3×3 zone spatial report for the LLM
                territory_trend: float — recent growth/shrink rate
        """
        self._frame_count += 1

        # Step 1: Capture raw screenshot from browser
        screenshot_bytes = await self.page.screenshot()
        raw_image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")

        # Step 2: Crop to game area (remove browser chrome)
        game_area = self._crop_game_area(raw_image)

        # Step 3: Estimate territory from color analysis
        territory_pct = self._estimate_territory(game_area)
        self._last_territory = territory_pct
        self._territory_history.append(territory_pct)

        # Step 4: Estimate number of players from distinct colors
        num_players = self._estimate_players(game_area)

        # Step 5: Generate the 3×3 grid report for the LLM
        grid_text = self.generate_grid_report(raw_image)

        # Step 6: Calculate trend
        territory_trend = self.get_territory_trend()

        return {
            "raw_image": raw_image,
            "territory_pct": territory_pct,
            "num_players": num_players,
            "game_area": game_area,
            "grid_text": grid_text,
            "territory_trend": territory_trend,
        }

    # ==================================================
    # GRID REPORT — 3×3 Zone Analysis for the LLM
    # ==================================================

    def generate_grid_report(self, image: Image.Image) -> str:
        """
        Divide the game screen into a 3×3 grid (9 zones) and analyze
        the color composition of each zone.

        For each zone, count Player (blue), Enemy (colored), Neutral (gray),
        and Ocean/Edge pixels, then produce a plain-English summary.

        Zone Layout:
            1=Top-Left    2=Top-Center    3=Top-Right
            4=Center-Left 5=Center        6=Center-Right
            7=Bot-Left    8=Bot-Center    9=Bot-Right

        Args:
            image: Full-resolution PIL Image (raw browser screenshot)

        Returns:
            str like:
            "Zone 1 (Top-Left): Mostly Neutral | Zone 2 (Top-Center): Heavy Enemy Presence | ..."
        """
        pixels = np.array(image, dtype=np.float32)
        img_h, img_w = pixels.shape[:2]

        # Game area boundaries (crop UI bars)
        y_min = GAME_AREA_TOP
        y_max = min(GAME_AREA_BOTTOM, img_h)
        x_min = GAME_AREA_LEFT
        x_max = min(GAME_AREA_RIGHT, img_w)

        game_h = y_max - y_min
        game_w = x_max - x_min

        zone_h = game_h // GRID_ROWS
        zone_w = game_w // GRID_COLS

        segments = []

        for zone_id in range(1, 10):
            # Convert zone_id (1-9) to row, col (0-indexed)
            row = (zone_id - 1) // GRID_COLS
            col = (zone_id - 1) % GRID_COLS

            # Pixel boundaries for this zone
            zy_start = y_min + row * zone_h
            zy_end = zy_start + zone_h
            zx_start = x_min + col * zone_w
            zx_end = zx_start + zone_w

            # Clamp
            zy_end = min(zy_end, img_h)
            zx_end = min(zx_end, img_w)

            # Extract zone pixels and downsample for speed
            zone_pixels = pixels[zy_start:zy_end, zx_start:zx_end]
            if zone_pixels.size == 0:
                segments.append(f"Zone {zone_id} ({ZONE_NAMES[zone_id]}): Unknown")
                continue

            # Downsample to ~32x32 for fast analysis
            step_y = max(1, zone_pixels.shape[0] // 32)
            step_x = max(1, zone_pixels.shape[1] // 32)
            sampled = zone_pixels[::step_y, ::step_x]

            # Classify every sampled pixel
            counts = {"Own Territory": 0, "Neutral": 0, "Enemy": 0, "Ocean/Edge": 0}
            for py in range(sampled.shape[0]):
                for px in range(sampled.shape[1]):
                    color = sampled[py, px].astype(np.int32)
                    label = self._classify_pixel(color)
                    counts[label] += 1

            total = sum(counts.values())
            if total == 0:
                segments.append(f"Zone {zone_id} ({ZONE_NAMES[zone_id]}): Unknown")
                continue

            # Calculate percentages
            pct_own = counts["Own Territory"] / total * 100
            pct_neutral = counts["Neutral"] / total * 100
            pct_enemy = counts["Enemy"] / total * 100
            pct_ocean = counts["Ocean/Edge"] / total * 100

            # Generate human-readable description
            desc = self._describe_zone(pct_own, pct_neutral, pct_enemy, pct_ocean)
            segments.append(f"Zone {zone_id} ({ZONE_NAMES[zone_id]}): {desc}")

        return " | ".join(segments)

    @staticmethod
    def _describe_zone(pct_own: float, pct_neutral: float, pct_enemy: float, pct_ocean: float) -> str:
        """
        Produce a concise plain-English description of a zone's composition.

        Args:
            pct_own:     % of zone that is our territory
            pct_neutral: % that is neutral gray land
            pct_enemy:   % that is enemy colored territory
            pct_ocean:   % that is ocean/edge/UI

        Returns:
            Description string like "Mostly Neutral" or "Heavy Enemy Presence"
        """
        # Dominant content
        if pct_own > 60:
            return "Our Territory (Safe)"
        if pct_ocean > 70:
            return "Ocean/Edge (Blocked)"
        if pct_neutral > 50:
            if pct_enemy > 15:
                return "Mostly Neutral, Some Enemy"
            return "Mostly Neutral (Free Land)"
        if pct_enemy > 50:
            return "Heavy Enemy Presence (Dangerous)"
        if pct_enemy > 25:
            if pct_neutral > 25:
                return "Mixed Enemy + Neutral"
            return "Enemy Territory (Risky)"
        if pct_own > 30:
            if pct_neutral > 20:
                return "Our Border + Neutral (Expand Here)"
            if pct_enemy > 15:
                return "Our Border + Enemy (Defend)"
            return "Our Territory Edge"

        # Fallback
        parts = []
        if pct_neutral > 20:
            parts.append(f"Neutral {pct_neutral:.0f}%")
        if pct_enemy > 10:
            parts.append(f"Enemy {pct_enemy:.0f}%")
        if pct_own > 10:
            parts.append(f"Ours {pct_own:.0f}%")
        if pct_ocean > 20:
            parts.append(f"Ocean {pct_ocean:.0f}%")
        return ", ".join(parts) if parts else "Mixed"

    @staticmethod
    def _classify_pixel(color: np.ndarray) -> str:
        """
        Classify a single pixel's RGB color into a game-meaningful label
        using HSV analysis.

        Classification Rules (Territorial.io specific):
            Own Territory:   Hue 200-260°, Sat >= 40%, Val >= 40%
            Neutral Land:    Sat < 15%, Val between 16-94%  (gray tones)
            Enemy Land:      Sat >= 20%, Val >= 20%  (vivid non-blue)
            Ocean/Edge:      Everything else (very dark, very bright, etc.)

        Args:
            color: numpy array [R, G, B] (int32)

        Returns:
            One of: "Own Territory", "Neutral", "Enemy", "Ocean/Edge"
        """
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

        h_deg = h * 360
        s_pct = s * 100
        v_pct = v * 100

        # Check for our own territory first (blue)
        if (PLAYER_HUE_MIN <= h_deg <= PLAYER_HUE_MAX
                and s_pct >= PLAYER_SAT_MIN
                and v_pct >= PLAYER_VAL_MIN):
            return "Own Territory"

        # Neutral land: low saturation, mid-range brightness (gray)
        if s_pct < 15 and 16 <= v_pct <= 94:
            return "Neutral"

        # Enemy land: saturated and visible (colored territory)
        if s_pct >= 20 and v_pct >= 20:
            return "Enemy"

        # Everything else: ocean, mountains, map edge, UI elements
        return "Ocean/Edge"

    @staticmethod
    def get_zone_center(zone_id: int) -> tuple:
        """
        Get the pixel coordinates of the center of a zone (1-9).

        Useful for game_environment.py to know where to scan
        when the LLM says "attack Zone X".

        Args:
            zone_id: Zone number 1-9

        Returns:
            (center_x, center_y) in full-image pixel coordinates
        """
        zone_id = max(1, min(9, zone_id))
        row = (zone_id - 1) // GRID_COLS
        col = (zone_id - 1) % GRID_COLS

        game_h = GAME_AREA_BOTTOM - GAME_AREA_TOP
        game_w = GAME_AREA_RIGHT - GAME_AREA_LEFT
        zone_h = game_h // GRID_ROWS
        zone_w = game_w // GRID_COLS

        cx = GAME_AREA_LEFT + col * zone_w + zone_w // 2
        cy = GAME_AREA_TOP + row * zone_h + zone_h // 2

        return (cx, cy)

    @staticmethod
    def get_zone_bounds(zone_id: int) -> tuple:
        """
        Get the pixel boundaries of a zone (1-9).

        Returns:
            (x_start, y_start, x_end, y_end) in full-image pixel coords
        """
        zone_id = max(1, min(9, zone_id))
        row = (zone_id - 1) // GRID_COLS
        col = (zone_id - 1) % GRID_COLS

        game_h = GAME_AREA_BOTTOM - GAME_AREA_TOP
        game_w = GAME_AREA_RIGHT - GAME_AREA_LEFT
        zone_h = game_h // GRID_ROWS
        zone_w = game_w // GRID_COLS

        x_start = GAME_AREA_LEFT + col * zone_w
        y_start = GAME_AREA_TOP + row * zone_h
        x_end = x_start + zone_w
        y_end = y_start + zone_h

        return (x_start, y_start, x_end, y_end)

    # ==================================================
    # TERRITORY & PLAYER ESTIMATION (Unchanged Logic)
    # ==================================================

    def _crop_game_area(self, image: Image.Image) -> Image.Image:
        """
        Crop the screenshot to the game map region, removing browser
        chrome, title bars, and game UI elements.

        Args:
            image: Full browser screenshot as PIL Image

        Returns:
            Cropped PIL Image containing only the game map
        """
        width, height = image.size

        left = min(GAME_AREA_LEFT, width)
        top = min(GAME_AREA_TOP, height)
        right = min(GAME_AREA_RIGHT, width)
        bottom = min(GAME_AREA_BOTTOM, height)

        if right <= left or bottom <= top:
            return image  # Return uncropped if boundaries are invalid

        return image.crop((left, top, right, bottom))

    def _estimate_territory(self, image: Image.Image) -> float:
        """
        Estimate the player's territory percentage by analyzing blue
        color coverage in the game screenshot.

        Uses vectorized numpy HSV conversion for performance instead
        of per-pixel Python loops.

        Args:
            image: Cropped game area PIL Image

        Returns:
            Float from 0.0 to 1.0 representing estimated territory %
        """
        # Downsample for speed
        small = image.resize((128, 128), Image.BILINEAR)
        pixels = np.array(small, dtype=np.float32) / 255.0

        total_pixels = pixels.shape[0] * pixels.shape[1]
        if total_pixels == 0:
            return 0.0

        # Vectorized HSV conversion using numpy
        r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue calculation (degrees)
        hue = np.zeros_like(cmax)
        mask_r = (cmax == r) & (delta > 0)
        mask_g = (cmax == g) & (delta > 0)
        mask_b = (cmax == b) & (delta > 0)
        hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
        hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
        hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

        # Saturation (percentage)
        sat = np.where(cmax > 0, (delta / cmax) * 100, 0)

        # Value (percentage)
        val = cmax * 100

        # Count blue territory pixels
        blue_mask = (
            (hue >= PLAYER_HUE_MIN) & (hue <= PLAYER_HUE_MAX)
            & (sat >= PLAYER_SAT_MIN)
            & (val >= PLAYER_VAL_MIN)
        )

        territory_pct = float(np.sum(blue_mask)) / total_pixels
        return max(0.0, min(1.0, territory_pct))

    def _estimate_players(self, image: Image.Image) -> int:
        """
        Estimate active player count by counting distinct territory
        colors (hue clusters) in the screenshot.

        Args:
            image: Cropped game area PIL Image

        Returns:
            Integer count of estimated active players (minimum 1)
        """
        small = image.resize((64, 64), Image.BILINEAR)
        pixels = np.array(small)

        hue_bins = Counter()
        for row in range(pixels.shape[0]):
            for col in range(pixels.shape[1]):
                r, g, b = pixels[row, col]
                h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                s_pct = s * 100
                v_pct = v * 100

                # Skip near-white, near-black, low-sat (neutral/water)
                if s_pct < 15 or v_pct < 15 or v_pct > 95:
                    continue

                hue_bin = int(h * 12)  # 30° buckets → 0-11
                hue_bins[hue_bin] += 1

        player_count = sum(1 for count in hue_bins.values() if count >= MIN_PLAYER_PIXELS)
        return max(1, player_count)

    def get_territory_trend(self, window: int = 10) -> float:
        """
        Get the recent territory trend (positive = growing, negative = shrinking).

        Args:
            window: Number of recent frames to analyze

        Returns:
            Float indicating territory change rate
        """
        if len(self._territory_history) < 2:
            return 0.0

        recent = self._territory_history[-window:]
        if len(recent) < 2:
            return 0.0

        return recent[-1] - recent[0]


# ==============================================
# STANDALONE UTILITY
# ==============================================

def estimate_territory_from_image(image: Image.Image) -> float:
    """
    Standalone territory estimation from a PIL Image.
    Useful for processing saved screenshots.

    Args:
        image: PIL Image of the game

    Returns:
        Float from 0.0 to 1.0
    """
    capture = ScreenCapture.__new__(ScreenCapture)
    capture._territory_history = []
    return capture._estimate_territory(image)


# ==============================================
# DEMO — Verify radar logic visually
# ==============================================

def demo():
    """Quick demo with a synthetic test image showing grid output."""
    print("\n📸 Screen Capture Demo — Grid Vision Edition")
    print("=" * 55)

    # Create a 1280x900 test image: gray background = neutral land
    test_img = Image.new("RGB", (1280, 900), (180, 180, 180))
    px = test_img.load()

    # Blue region at center (Zone 5) = our territory
    for x in range(480, 800):
        for y in range(300, 600):
            px[x, y] = (30, 80, 200)

    # Red enemy to the right (Zone 6) 
    for x in range(860, 1200):
        for y in range(300, 600):
            px[x, y] = (200, 40, 40)

    # Green enemy bottom-center (Zone 8)
    for x in range(480, 800):
        for y in range(620, 830):
            px[x, y] = (40, 180, 60)

    # Dark ocean top row (Zone 1, 2, 3)
    for x in range(0, 1280):
        for y in range(0, 80):
            px[x, y] = (10, 15, 30)

    # Process
    capture = ScreenCapture.__new__(ScreenCapture)
    capture._frame_count = 0
    capture._last_territory = 0.0
    capture._territory_history = []

    game_area = capture._crop_game_area(test_img)
    territory = capture._estimate_territory(game_area)
    players = capture._estimate_players(game_area)
    grid = capture.generate_grid_report(test_img)

    print(f"  🖼️  Test image size: {test_img.size}")
    print(f"  🎮 Game area size:  {game_area.size}")
    print(f"  🗺️  Territory:       {territory:.2%}")
    print(f"  👥 Players:         {players}")
    print(f"\n  📡 GRID REPORT:")
    for segment in grid.split(" | "):
        print(f"     → {segment}")

    # Test zone coordinate helpers
    print(f"\n  📍 Zone Centers:")
    for z in range(1, 10):
        cx, cy = ScreenCapture.get_zone_center(z)
        print(f"     Zone {z} ({ZONE_NAMES[z]}): ({cx}, {cy})")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
