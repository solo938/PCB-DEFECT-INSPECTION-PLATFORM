# src/utils/config.py
"""
Centralized configuration for the entire project.
All constants used by multiple modules live here.

No functions, no classes, no mkdir(), no logging.
Only constants.
"""

# ─────────────────────────────────────────────
# Dataset Information
# ─────────────────────────────────────────────

DATASET_NAME = "DeepPCB"
EXPECTED_PAIRS = 1500
EXPECTED_IMAGE_SIZE = (640, 640)
MIN_VALID_PAIRS = 1400


# ─────────────────────────────────────────────
# File Suffixes
# ─────────────────────────────────────────────

TEST_SUFFIX = "_test.jpg"
TEMP_SUFFIX = "_temp.jpg"
ANNOTATION_SUFFIX = ".txt"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


# ─────────────────────────────────────────────
# Class Mappings
# ─────────────────────────────────────────────

# DeepPCB uses 1-indexed class IDs.
# convert.py remaps these to 0-indexed YOLO format.
# Do not change these values without updating convert.py.

CLASS_ID_TO_NAME = {
    1: "open_circuit",
    2: "short",
    3: "mouse_bite",
    4: "spur",
    5: "spurious_copper",
    6: "pin_hole",
}

CLASS_NAME_TO_ID = {
    "open_circuit": 1,
    "short": 2,
    "mouse_bite": 3,
    "spur": 4,
    "spurious_copper": 5,
    "pin_hole": 6,
}

VALID_CLASS_IDS = set(CLASS_ID_TO_NAME.keys())
NUM_CLASSES = len(VALID_CLASS_IDS)

# YOLO format: 0-indexed, sorted by class ID
YOLO_CLASS_ID_TO_NAME = {
    i: CLASS_ID_TO_NAME[cid]
    for i, cid in enumerate(sorted(VALID_CLASS_IDS))
}

YOLO_CLASS_NAME_TO_ID = {
    name: idx
    for idx, name in YOLO_CLASS_ID_TO_NAME.items()
}


# ─────────────────────────────────────────────
# Split Configuration
# ─────────────────────────────────────────────

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42


# ─────────────────────────────────────────────
# Augmentation Configuration
# ─────────────────────────────────────────────

AUG_COPIES_PER_IMAGE = 3
AUG_FLIP_PROB = 0.5
AUG_ROTATE_PROB = 0.5
AUG_BRIGHTNESS_PROB = 0.5
AUG_NOISE_PROB = 0.3
AUG_MOTION_BLUR_PROB = 0.2
AUG_CLAHE_PROB = 0.3


# ─────────────────────────────────────────────
# Training Defaults
# ─────────────────────────────────────────────

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 4
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005


# ─────────────────────────────────────────────
# Validation Thresholds
# ─────────────────────────────────────────────

FAILURE_RATE_THRESHOLD = 0.1  # 10% max failures before pipeline halts
MIN_BBOXES_PER_CLASS = 50     # Minimum samples per class for training


# ─────────────────────────────────────────────
# Inference Defaults
# ─────────────────────────────────────────────

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 300


# ─────────────────────────────────────────────
# Camera / Video
# ─────────────────────────────────────────────

DEFAULT_CAMERA_ID = 0
DEFAULT_FRAME_INTERVAL = 30  # Process every Nth frame
DEFAULT_RESIZE_WIDTH = 640
DEFAULT_RESIZE_HEIGHT = 640