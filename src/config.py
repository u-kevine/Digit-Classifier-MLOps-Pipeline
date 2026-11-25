"""Central configuration. Every path can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"            # new labelled digits waiting for retraining

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
MODEL_PATH = MODEL_DIR / "digit_cnn.h5"
META_PATH = MODEL_DIR / "model_meta.json"      # class names, metrics, version history
STATUS_PATH = MODEL_DIR / "retrain_status.json"
ARCHIVE_DIR = MODEL_DIR / "archive"

IMG_SIZE = (32, 32)        # digits are 8x8 stored as 64x64; 32x32 keeps every detail
CHANNELS = 1               # grayscale
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
SEED = 42
VAL_SPLIT = 0.2

RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", 50))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
PROMOTION_TOLERANCE = float(os.getenv("PROMOTION_TOLERANCE", 0.01))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

for d in (TRAIN_DIR, TEST_DIR, UPLOAD_DIR, MODEL_DIR, ARCHIVE_DIR):
    d.mkdir(parents=True, exist_ok=True)