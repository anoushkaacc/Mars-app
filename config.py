"""Paths, Hugging Face checkpoint source, class labels and preprocessing settings."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = APP_DIR / "checkpoints"

# The supplied Hugging Face URL is a DATASET repository, not a Model repository.
HF_REPO_ID = "Anoushkaacc/mars-segmentation-checkpoints"
HF_REPO_TYPE = "dataset"

# Use the full local validation directory when it exists. On Streamlit Cloud,
# the Windows D: path does not exist, so the app falls back to sample_images/.
LOCAL_TEST_IMAGE_DIR = Path(r"D:\mars_dataset\val\images")
SAMPLE_IMAGE_DIR = APP_DIR / "sample_images"
TEST_IMAGE_DIR = (
    LOCAL_TEST_IMAGE_DIR if LOCAL_TEST_IMAGE_DIR.is_dir() else SAMPLE_IMAGE_DIR
)

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

# Twelve-class names and palette in the class-ID order provided.
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
    (0, 0, 0),          # Unknown / background
    (165, 42, 42),      # Rough terrain
    (128, 128, 0),      # Dunes
    (255, 192, 203),    # Smooth terrain
    (255, 255, 0),      # Channel
    (0, 255, 0),        # Bedrock
    (255, 0, 255),     # Cliff
    (255, 165, 0),      # Impact ejecta
    (0, 255, 255),     # Crater
    (128, 0, 128),      # Ridge
    (0, 0, 255),        # Boulder
    (255, 0, 0),        # Alluvial fan
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

assert len(FIVE_CLASS_NAMES) == len(FIVE_CLASS_PALETTE) == 5
assert len(TWELVE_CLASS_NAMES) == len(TWELVE_CLASS_PALETTE) == 12
