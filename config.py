"""
config.py
=========
Central configuration for the Yoga Pose Detection & Correction project.

Single source of truth for:
    - File / directory paths
    - The 6 target yoga poses and their dataset statistics
    - Model hyper-parameters (EfficientNetB0 primary, MobileNetV2 backup)
    - Image settings
    - MediaPipe settings
    - Pose-correction feedback thresholds

Every other module imports from here so configuration stays consistent.
"""

import os

# ===================================================================
# 1. PATHS
# ===================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset root — must contain one sub-folder per pose.
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Cleaned dataset (after corrupted / duplicate removal).
CLEAN_DATASET_DIR = os.path.join(BASE_DIR, "dataset_clean")

# Model & artifact output directory.
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Primary model (will be auto-selected as best across all architectures).
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")

# Best model path (comparison winner, saved after train_best.py).
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_yoga_model.keras")

# Backup model (MobileNetV2).
MODEL_BACKUP_PATH = os.path.join(MODEL_DIR, "best_model_mobilenetv2.keras")

# Landmark-only model.
MODEL_LANDMARK_PATH = os.path.join(MODEL_DIR, "best_model_landmarks.keras")

# Label mapping (index -> pose name).
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")

# Training history (JSON, for the dashboard).
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

# Evaluation outputs (saved into OUTPUT_DIR).
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
TRAINING_ACCURACY_PATH = os.path.join(OUTPUT_DIR, "training_accuracy.png")
TRAINING_HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.png")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")
EVALUATION_METRICS_PATH = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")

# MediaPipe pose-landmarker model file.
POSE_LANDMARKER_PATH = os.path.join(BASE_DIR, "pose_landmarker.task")
POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# Ensure directories exist.
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================================
# 2. ALL YOGA POSES (17 classes)
# ===================================================================
# Canonical pose names (must match dataset sub-folder names, case-insensitive).
POSES = [
    "Boat Pose",
    "Bridge Pose",
    "Camel Pose",
    "Cat Pose",
    "Chair Pose",
    "Child Pose",
    "Cobra Pose",
    "Corpse Pose",
    "Cow Pose",
    "Cross Legged Pose",
    "Downward Dog",
    "Goddess Pose",
    "Mountain Pose",
    "Plank Pose",
    "Tree Pose",
    "Triangle Pose",
    "Warrior 2",
]

NUM_CLASSES = len(POSES)  # 17

# Expected dataset statistics (for validation / reporting).
DATASET_STATS = {
    "Boat Pose": 68,
    "Bridge Pose": 58,
    "Camel Pose": 87,
    "Cat Pose": 46,
    "Chair Pose": 81,
    "Child Pose": 71,
    "Cobra Pose": 73,
    "Corpse Pose": 57,
    "Cow Pose": 94,
    "Cross Legged Pose": 50,
    "Downward Dog": 320,
    "Goddess Pose": 260,
    "Mountain Pose": 56,
    "Plank Pose": 381,
    "Tree Pose": 228,
    "Triangle Pose": 191,
    "Warrior 2": 361,
}
TOTAL_IMAGES = sum(DATASET_STATS.values())  # 2482

# Mapping of common alternative folder names -> canonical names.
POSE_ALIASES = {
    # --- Boat Pose ---
    "boat": "Boat Pose",
    "boatpose": "Boat Pose",
    "navasana": "Boat Pose",
    # --- Bridge Pose ---
    "bridge": "Bridge Pose",
    "bridgepose": "Bridge Pose",
    "bridge psoe": "Bridge Pose",
    "setu bandhasana": "Bridge Pose",
    # --- Camel Pose ---
    "camel": "Camel Pose",
    "camelpose": "Camel Pose",
    "ustrasana": "Camel Pose",
    # --- Cat Pose ---
    "cat": "Cat Pose",
    "catpose": "Cat Pose",
    "marjaryasana": "Cat Pose",
    # --- Chair Pose ---
    "chair": "Chair Pose",
    "chairpose": "Chair Pose",
    "utkatasana": "Chair Pose",
    # --- Child Pose ---
    "child": "Child Pose",
    "childpose": "Child Pose",
    "childs pose": "Child Pose",
    "balasana": "Child Pose",
    # --- Cobra Pose ---
    "cobra": "Cobra Pose",
    "cobrapose": "Cobra Pose",
    "bhujangasana": "Cobra Pose",
    # --- Corpse Pose ---
    "corpse": "Corpse Pose",
    "corpsepose": "Corpse Pose",
    "savasana": "Corpse Pose",
    # --- Cow Pose ---
    "cow": "Cow Pose",
    "cowpose": "Cow Pose",
    "bitilasana": "Cow Pose",
    # --- Cross Legged Pose ---
    "crosslegged": "Cross Legged Pose",
    "cross_legged": "Cross Legged Pose",
    "cross legged": "Cross Legged Pose",
    "crossleggedpose": "Cross Legged Pose",
    "sukhasana": "Cross Legged Pose",
    # --- Downward Dog ---
    "downdog": "Downward Dog",
    "downdogpose": "Downward Dog",
    "downtwarddog": "Downward Dog",
    "downwarddog": "Downward Dog",
    "adhomukhasvanasana": "Downward Dog",
    # --- Goddess Pose ---
    "goddess": "Goddess Pose",
    "goddesspose": "Goddess Pose",
    "deviasana": "Goddess Pose",
    # --- Mountain Pose ---
    "mountain": "Mountain Pose",
    "mountainpose": "Mountain Pose",
    "tadasana": "Mountain Pose",
    # --- Plank Pose ---
    "plank": "Plank Pose",
    "plankpose": "Plank Pose",
    "phalakasana": "Plank Pose",
    "kumbhakasana": "Plank Pose",
    # --- Tree Pose ---
    "tree": "Tree Pose",
    "treepose": "Tree Pose",
    "vrksasana": "Tree Pose",
    # --- Triangle Pose ---
    "triangle": "Triangle Pose",
    "trianglepose": "Triangle Pose",
    "trikonasana": "Triangle Pose",
    # --- Warrior 2 ---
    "warrior2": "Warrior 2",
    "warriorii": "Warrior 2",
    "warrior2pose": "Warrior 2",
    "warrior 2": "Warrior 2",
    "virabhadrasana2": "Warrior 2",
    "virabhadrasanaii": "Warrior 2",
}

# ===================================================================
# 3. IMAGE SETTINGS
# ===================================================================
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ===================================================================
# 4. MODEL HYPER-PARAMETERS
# ===================================================================
LEARNING_RATE = 0.001            # Phase-1 (frozen base)
FINE_TUNE_LEARNING_RATE = 1e-5   # Phase-2 (fine-tuning)
EPOCHS = 100                     # Max epochs per phase (early stopping applies)
FINE_TUNE_EPOCHS = 50            # Max fine-tuning epochs
BATCH_SIZE = 32

# Train / validation / test split ratios.
VAL_SPLIT = 0.15                 # 15 % validation
TEST_SPLIT = 0.15                # 15 % test  (held out, evaluated at end)
RANDOM_SEED = 42                 # Reproducible splits

# Fine-tuning: unfreeze from this layer index onward.
FINE_TUNE_UNFREEZE_FROM = 100

# Regularisation.
LABEL_SMOOTHING = 0.1
DROPOUT_1 = 0.5                  # After first Dense layer
DROPOUT_2 = 0.4                  # After second Dense layer
L2_REG = 1e-4

# Early stopping / LR scheduler.
EARLY_STOP_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-7

# ===================================================================
# 5. FEEDBACK / CONFIDENCE THRESHOLDS
# ===================================================================
CONF_HIGH = 0.80                 # >= 80 % → great
CONF_MEDIUM = 0.60               # 60–79 % → almost there
# < 60 % → try again

DEFAULT_CONF_THRESHOLD = 0.50    # Default slider value in the Streamlit UI.

# Angle deviation tolerance (degrees) for pose correction feedback.
ANGLE_TOLERANCE = 15.0           # Within ±15° → "good"

# ===================================================================
# 6. MEDIAPIPE SETTINGS
# ===================================================================
MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE = 0.5

# Pose landmarker model quality: "lite" (fastest) | "full" (balanced) | "heavy" (most accurate)
POSE_MODEL_QUALITY = "full"

# Number of pose landmarks from MediaPipe (33 landmarks).
NUM_LANDMARKS = 33
LANDMARK_DIM = 4                 # x, y, z, visibility
LANDMARK_FEATURE_SIZE = NUM_LANDMARKS * LANDMARK_DIM  # 132

# ===================================================================
# 7. HELPER
# ===================================================================
def print_config():
    """Print a readable summary of the current configuration."""
    print("=" * 60)
    print("  YOGA POSE DETECTION & CORRECTION — CONFIG")
    print("=" * 60)
    print(f"  Base dir          : {BASE_DIR}")
    print(f"  Dataset dir       : {DATASET_DIR}")
    print(f"  Clean dataset dir : {CLEAN_DATASET_DIR}")
    print(f"  Model path        : {MODEL_PATH}")
    print(f"  Backup model path : {MODEL_BACKUP_PATH}")
    print(f"  Labels path       : {LABELS_PATH}")
    print(f"  Output dir        : {OUTPUT_DIR}")
    print(f"  Num classes       : {NUM_CLASSES}")
    print(f"  Poses             : {', '.join(POSES)}")
    print(f"  Total images      : {TOTAL_IMAGES}")
    print(f"  Image size        : {IMG_SIZE}")
    print(f"  Batch size        : {BATCH_SIZE}")
    print(f"  Learning rate     : {LEARNING_RATE}")
    print(f"  Fine-tune LR      : {FINE_TUNE_LEARNING_RATE}")
    print(f"  Epochs (phase 1)  : {EPOCHS}")
    print(f"  Epochs (phase 2)  : {FINE_TUNE_EPOCHS}")
    print(f"  Val / Test split  : {VAL_SPLIT} / {TEST_SPLIT}")
    print(f"  Label smoothing   : {LABEL_SMOOTHING}")
    print(f"  Dropout 1 / 2     : {DROPOUT_1} / {DROPOUT_2}")
    print(f"  Early stop patience: {EARLY_STOP_PATIENCE}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
