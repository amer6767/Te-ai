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

# ---- Radar Configuration ----
RADAR_CENTER_X = 640      # Center of 1280-wide game view
RADAR_CENTER_Y = 450      # Center of 900-tall game view (approx)
RADAR_MAX_DISTANCE = 400  # Maximum ray length in pixels
RADAR_STEP_SIZE = 5       # Pixels per step along each ray
RADAR_COLOR_THRESHOLD = 1000  # Squared color diff to detect a border

# 8 compass directions: (name, angle_in_degrees)
# Angles: 0°=East(right), 90°=South(down), 180°=West(left), 270°=North(up)
RADAR_DIRECTIONS = [
    ("North",      270),
    ("North-East", 315),
    ("East",         0),
    ("South-East",  45),
    ("South",       90),
    ("South-West", 135),
    ("West",       180),
    ("North-West", 225),
]


# ==============================================
# SCREEN CAPTURE CLASS
# ==============================================

class ScreenCapture:
    """
    Processes Playwright browser screenshots for the AI system.

    Provides:
    - Text Radar Report: 8-direction spatial scan for Nemotron LLM
    - Territory percentage estimation from blue-pixel analysis
    - Player count estimation from distinct hue clusters
    - Territory trend tracking across frames

    Usage:
        capture = ScreenCapture(page)
        frame = await capture.get_processed_frame()
        radar  = frame["radar_text"]       # Nemotron reads this
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
                raw_image:     PIL Image — full browser screenshot
                territory_pct: float — estimated player territory % (0.0–1.0)
                num_players:   int — estimated number of distinct players
                game_area:     PIL Image — cropped to game map only
                radar_text:    str — 8-direction spatial report for Nemotron
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

        # Step 5: Generate the text radar report for Nemotron
        radar_text = self.generate_radar_report(raw_image)

        # Step 6: Calculate trend
        territory_trend = self.get_territory_trend()

        return {
            "raw_image": raw_image,
            "territory_pct": territory_pct,
            "num_players": num_players,
            "game_area": game_area,
            "radar_text": radar_text,
            "territory_trend": territory_trend,
        }

    # ==================================================
    # RADAR REPORT — The Core Intelligence for Nemotron
    # ==================================================

    def generate_radar_report(self, image: Image.Image) -> str:
        """
        Cast 8 directional rays from the screen center outward,
        classify what each ray hits (Neutral, Enemy, Own, Ocean/Edge),
        measure the distance, and return a single plain-English string
        that Nemotron can reason over.

        The ray walks outward in RADAR_STEP_SIZE pixel increments.
        When the squared RGB difference from the center pixel exceeds
        RADAR_COLOR_THRESHOLD, a border is detected. The pixel at the
        border is then classified via its HSV values.

        Args:
            image: Full-resolution PIL Image (raw browser screenshot)

        Returns:
            str like:
            "North: Enemy (Very Close) | East: Neutral (Medium) | ..."
        """
        pixels = np.array(image, dtype=np.int32)  # int32 avoids uint8 overflow in diff calc
        img_h, img_w = pixels.shape[:2]

        cx = min(RADAR_CENTER_X, img_w - 1)
        cy = min(RADAR_CENTER_Y, img_h - 1)

        # Reference color at exact center (our territory color)
        player_color = pixels[cy, cx].astype(np.int32)

        segments = []

        for direction_name, angle_deg in RADAR_DIRECTIONS:
            angle_rad = math.radians(angle_deg)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)

            hit_distance = RADAR_MAX_DISTANCE  # default if we reach max range
            hit_color = None
            hit_type = "Ocean/Edge"

            # Walk the ray outward step by step
            for step in range(1, (RADAR_MAX_DISTANCE // RADAR_STEP_SIZE) + 1):
                px = int(cx + dx * step * RADAR_STEP_SIZE)
                py = int(cy + dy * step * RADAR_STEP_SIZE)

                # Bounds check — if we leave the screen, it's an edge
                if px < 0 or px >= img_w or py < 0 or py >= img_h:
                    hit_distance = step * RADAR_STEP_SIZE
                    hit_type = "Ocean/Edge"
                    break

                current_color = pixels[py, px].astype(np.int32)

                # Squared color difference from center
                diff = int(np.sum((current_color - player_color) ** 2))

                if diff > RADAR_COLOR_THRESHOLD:
                    # Border detected! Record distance and classify.
                    hit_distance = step * RADAR_STEP_SIZE
                    hit_color = current_color
                    break

            # Classify what we hit
            if hit_color is not None:
                hit_type = self._classify_pixel(hit_color)

            # Convert pixel distance to human-readable word
            distance_label = self._distance_label(hit_distance)

            # Build this direction's segment string
            if hit_type == "Ocean/Edge":
                segments.append(f"{direction_name}: Ocean/Edge")
            else:
                segments.append(f"{direction_name}: {hit_type} ({distance_label})")

        return " | ".join(segments)

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
    def _distance_label(distance_px: int) -> str:
        """
        Convert a pixel distance into a Nemotron-friendly word.

        Thresholds are tuned for 1280x900 game view:
            < 50px  → Very Close  (immediate border, attack NOW)
            < 150px → Medium      (within comfortable reach)
            >= 150px → Far        (long march, consider economy first)

        Args:
            distance_px: Distance in pixels from screen center

        Returns:
            "Very Close", "Medium", or "Far"
        """
        if distance_px < 50:
            return "Very Close"
        elif distance_px < 150:
            return "Medium"
        else:
            return "Far"

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
    """Quick demo with a synthetic test image showing radar output."""
    print("\n📸 Screen Capture Demo — Text Radar Edition")
    print("=" * 55)

    # Create a 1280x900 test image: gray background = neutral land
    test_img = Image.new("RGB", (1280, 900), (180, 180, 180))
    px = test_img.load()

    # Blue region at center = our territory
    for x in range(540, 740):
        for y in range(350, 550):
            px[x, y] = (30, 80, 200)

    # Red enemy to the East
    for x in range(800, 1000):
        for y in range(350, 550):
            px[x, y] = (200, 40, 40)

    # Green enemy to the South
    for x in range(540, 740):
        for y in range(600, 750):
            px[x, y] = (40, 180, 60)

    # Dark ocean in the North
    for x in range(0, 1280):
        for y in range(0, 100):
            px[x, y] = (10, 15, 30)

    # Process
    capture = ScreenCapture.__new__(ScreenCapture)
    capture._frame_count = 0
    capture._last_territory = 0.0
    capture._territory_history = []

    game_area = capture._crop_game_area(test_img)
    territory = capture._estimate_territory(game_area)
    players = capture._estimate_players(game_area)
    radar = capture.generate_radar_report(test_img)

    print(f"  🖼️  Test image size: {test_img.size}")
    print(f"  🎮 Game area size:  {game_area.size}")
    print(f"  🗺️  Territory:       {territory:.2%}")
    print(f"  👥 Players:         {players}")
    print(f"\n  📡 RADAR REPORT:")
    for segment in radar.split(" | "):
        print(f"     → {segment}")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
