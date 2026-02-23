"""
=============================================================================
screen_capture.py — Browser Screenshot Processing for Territorial.io AI
=============================================================================

Processes raw browser screenshots into usable data:
- Tensors for the CNN (brain.py)
- Territory percentage estimates from color analysis
- Player count estimates from distinct colors
- Cropped game area with browser UI removed

This replaces all random.uniform and random.randint placeholders in
trainer.py _score_move with real values from get_processed_frame().

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
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

# CNN input size matching brain.py Config
CNN_INPUT_WIDTH = 128
CNN_INPUT_HEIGHT = 128

# Game area boundaries (approximate, browser UI removed)
# These define the rectangular region of the actual game map
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
NEUTRAL_HUE_MIN = 0
NEUTRAL_SAT_MAX = 15     # Very low saturation = gray

# Minimum distinct pixels to count as a player color
MIN_PLAYER_PIXELS = 100

# Image preprocessing pipeline matching brain.py GamePreprocessor
TRANSFORM = T.Compose([
    T.Resize((CNN_INPUT_HEIGHT, CNN_INPUT_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==============================================
# SCREEN CAPTURE CLASS
# ==============================================

class ScreenCapture:
    """
    Processes Playwright browser screenshots for the AI system.
    
    Provides:
    - Processed frames with tensor, territory %, player count
    - Game area cropping to remove browser UI
    - Color-based territory estimation
    - Player count estimation from distinct territory colors
    
    Usage:
        capture = ScreenCapture(page)
        frame = await capture.get_processed_frame()
        tensor = frame["tensor"]           # Ready for brain.py CNN
        territory = frame["territory_pct"] # Our territory percentage
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
                tensor:        torch.Tensor — 128x128 processed for brain.py CNN
                territory_pct: float — estimated player territory percentage (0.0–1.0)
                num_players:   int — estimated number of distinct players
                game_area:     PIL Image — cropped to game map only
        """
        self._frame_count += 1

        # Step 1: Capture raw screenshot from browser
        screenshot_bytes = await self.page.screenshot()
        raw_image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")

        # Step 2: Crop to game area (remove browser chrome)
        game_area = self._crop_game_area(raw_image)

        # Step 3: Convert to tensor for CNN
        tensor = TRANSFORM(game_area).unsqueeze(0)  # Add batch dimension [1, 3, 128, 128]

        # Step 4: Estimate territory from color analysis
        territory_pct = self._estimate_territory(game_area)
        self._last_territory = territory_pct
        self._territory_history.append(territory_pct)

        # Step 5: Estimate number of players from distinct colors
        num_players = self._estimate_players(game_area)

        return {
            "raw_image": raw_image,
            "tensor": tensor,
            "territory_pct": territory_pct,
            "num_players": num_players,
            "game_area": game_area,
        }

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

        # Calculate crop boundaries, clamped to image dimensions
        left = min(GAME_AREA_LEFT, width)
        top = min(GAME_AREA_TOP, height)
        right = min(GAME_AREA_RIGHT, width)
        bottom = min(GAME_AREA_BOTTOM, height)

        # Ensure valid crop region
        if right <= left or bottom <= top:
            return image  # Return uncropped if boundaries are invalid

        cropped = image.crop((left, top, right, bottom))
        return cropped

    def _estimate_territory(self, image: Image.Image) -> float:
        """
        Estimate the player's territory percentage by analyzing blue
        color coverage in the game screenshot.
        
        Territorial.io uses blue for the local player's territory.
        We count blue pixels vs total game area pixels.
        
        Args:
            image: Cropped game area PIL Image
            
        Returns:
            Float from 0.0 to 1.0 representing estimated territory %
        """
        # Downsample for speed (analysis doesn't need full resolution)
        small = image.resize((128, 128), Image.BILINEAR)
        pixels = np.array(small)

        total_pixels = pixels.shape[0] * pixels.shape[1]
        if total_pixels == 0:
            return 0.0

        # Convert RGB to HSV for better color detection
        blue_count = 0
        for row in range(0, pixels.shape[0], 2):  # Skip every other row for speed
            for col in range(0, pixels.shape[1], 2):
                r, g, b = pixels[row, col]
                # Convert to HSV
                h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                h_deg = h * 360
                s_pct = s * 100
                v_pct = v * 100

                # Check if pixel is in player's blue range
                if (PLAYER_HUE_MIN <= h_deg <= PLAYER_HUE_MAX and
                        s_pct >= PLAYER_SAT_MIN and
                        v_pct >= PLAYER_VAL_MIN):
                    blue_count += 1

        # We sampled every other pixel in both dimensions = 1/4 of pixels
        sampled_pixels = (pixels.shape[0] // 2) * (pixels.shape[1] // 2)
        if sampled_pixels == 0:
            return 0.0

        territory_pct = blue_count / sampled_pixels
        # Clamp to valid range
        return max(0.0, min(1.0, territory_pct))

    def _estimate_players(self, image: Image.Image) -> int:
        """
        Estimate the number of active players by counting distinct
        territory colors in the screenshot.
        
        Each player in Territorial.io has a unique color. We bucket
        pixel hues and count how many distinct hue clusters exist.
        
        Args:
            image: Cropped game area PIL Image
            
        Returns:
            Integer count of estimated active players (minimum 1)
        """
        # Downsample heavily for clustering speed
        small = image.resize((64, 64), Image.BILINEAR)
        pixels = np.array(small)

        # Bin pixels by hue (30-degree buckets = 12 bins)
        hue_bins = Counter()
        for row in range(pixels.shape[0]):
            for col in range(pixels.shape[1]):
                r, g, b = pixels[row, col]
                h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                s_pct = s * 100
                v_pct = v * 100

                # Skip near-white, near-black, and low-saturation (neutral/water)
                if s_pct < 15 or v_pct < 15 or v_pct > 95:
                    continue

                # Bucket hue into 30-degree bins
                hue_bin = int(h * 12)  # 0-11
                hue_bins[hue_bin] += 1

        # Count bins with significant pixel counts as distinct players
        player_count = 0
        for hue_bin, count in hue_bins.items():
            if count >= MIN_PLAYER_PIXELS:
                player_count += 1

        # At least 1 player (us)
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
# UTILITY FUNCTIONS
# ==============================================

def process_pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a tensor ready for brain.py's CNN.
    Standalone utility function for use outside the class.
    
    Args:
        image: PIL Image (any size)
        
    Returns:
        Tensor of shape [1, 3, 128, 128]
    """
    return TRANSFORM(image).unsqueeze(0)


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
    return capture._estimate_territory(image)


# ==============================================
# DEMO
# ==============================================

def demo():
    """Quick demo with a synthetic test image."""
    print("\n📸 Screen Capture Demo")
    print("=" * 50)

    # Create a test image with some blue territory
    test_img = Image.new("RGB", (1280, 900), (200, 200, 200))
    pixels = test_img.load()

    # Add a blue region (player territory)
    for x in range(200, 500):
        for y in range(200, 400):
            pixels[x, y] = (30, 80, 200)  # Blue

    # Add a red region (enemy territory)
    for x in range(600, 800):
        for y in range(300, 500):
            pixels[x, y] = (200, 40, 40)  # Red

    # Add a green region (another enemy)
    for x in range(400, 550):
        for y in range(500, 650):
            pixels[x, y] = (40, 180, 60)  # Green

    # Process the test image
    capture = ScreenCapture.__new__(ScreenCapture)
    capture._frame_count = 0
    capture._last_territory = 0.0
    capture._territory_history = []

    game_area = capture._crop_game_area(test_img)
    territory = capture._estimate_territory(game_area)
    players = capture._estimate_players(game_area)

    tensor = process_pil_to_tensor(game_area)

    print(f"  🖼️  Test image size: {test_img.size}")
    print(f"  🎮 Game area size: {game_area.size}")
    print(f"  🧠 Tensor shape: {tensor.shape}")
    print(f"  🗺️  Estimated territory: {territory:.2%}")
    print(f"  👥 Estimated players: {players}")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
