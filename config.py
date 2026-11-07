"""Local paths, class labels and preprocessing settings for the Mars viewer."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# Put the five supported checkpoints in D:\\mars_app\\checkpoints.
# Since this file sits in D:\\mars_app, the path resolves automatically.
CHECKPOINT_DIR = APP_DIR / "checkpoints"

# The user-provided unlabelled test-image folder. Change this only if the folder moves.
TEST_IMAGE_DIR = Path(r"D:\mars_dataset\val\images")

# Five-class palette from the supplied ynet-class-weights notebook.
# The first five colors in Dynamic_MARTANET are the same; its sample notebook
# additionally defines Boulders as a sixth class, while unet_cbam.pth has 5 outputs.
FIVE_CLASS_NAMES = ["Background", "Crater", "Rough", "Smooth", "Alluvial Fan"]
FIVE_CLASS_PALETTE = [
    (0, 0, 0),
    (255, 255, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]

# Twelve-class CCHNet palette, copied in class-ID order from all three supplied
# CCHNet training scripts. Those scripts do not provide semantic terrain names.
TWELVE_CLASS_NAMES = [f"Class {i}" for i in range(12)]
TWELVE_CLASS_PALETTE = [
    (0, 0, 0), #background
    (165, 42, 42), #rough terrain
    (128, 128, 0), #dunes
    (255, 192, 203), #smooth terrain
    (255, 255, 0), #channel
    (0, 255, 0), #bedrock
    (255, 0, 255), #cliff 
    (255, 165, 0), #impact ejecta
    (0, 255, 255), #crater
    (128, 0, 128), #ridge
    (0, 0, 255), #boulder
    (255, 0, 0), #alluvial fan
]

# ImageNet normalization is used by the supplied CCHNet pipelines only.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
