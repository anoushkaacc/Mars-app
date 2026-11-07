"""Local paths, class labels, palettes and preprocessing settings."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = APP_DIR / "checkpoints"

# Interpreted from the path supplied in the request: D:\\mars_dataset\\val\\images
TEST_IMAGE_DIR = Path(r"D:\mars_dataset\val\images")

# Five-class palette used by the supplied five-class U-Net / MarTA-Net models.
FIVE_CLASS_NAMES = [
    "Background",
    "Crater",
    "Rough",
    "Smooth",
    "Alluvial Fan",
]
FIVE_CLASS_PALETTE = [
    (0, 0, 0),          # Background
    (255, 255, 0),      # Crater
    (255, 0, 0),        # Rough
    (0, 255, 0),        # Smooth
    (0, 0, 255),        # Alluvial Fan
]

# Twelve-class palette and names in the class-ID order supplied by the user.
TWELVE_CLASS_NAMES = [
    "Unknown",
    "Rough terrain",
    "Dunes",
    "Smooth terrain",
    "Channel",
    "Bedrock",
    "Cliff",
    "Impact ejecta",
    "Crater",
    "Ridge",
    "Boulder",
    "Alluvial fan",
]
TWELVE_CLASS_PALETTE = [
    (0, 0, 0),          # Background
    (165, 42, 42),      # Rough terrain
    (128, 128, 0),      # Dunes
    (255, 192, 203),    # Smooth terrain
    (255, 255, 0),      # Channel
    (0, 255, 0),        # Bedrock
    (255, 0, 255),      # Cliff
    (255, 165, 0),      # Impact ejecta
    (0, 255, 255),     # Crater
    (128, 0, 128),      # Ridge
    (0, 0, 255),        # Boulder
    (255, 0, 0),        # Alluvial fan
]

# ImageNet normalization is used by the supplied CCHNet pipelines only.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Fail early if a palette/name list drifts out of alignment with model outputs.
assert len(FIVE_CLASS_NAMES) == len(FIVE_CLASS_PALETTE) == 5
assert len(TWELVE_CLASS_NAMES) == len(TWELVE_CLASS_PALETTE) == 12
