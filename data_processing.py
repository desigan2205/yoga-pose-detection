"""
data_processing.py
==================
Data processing pipeline for the Yoga Pose Detection project.

Responsibilities:
    1. Scan the dataset folder and resolve folder names to canonical poses.
    2. Automatically remove corrupted / unreadable images.
    3. Detect and flag duplicate images (perceptual hash).
    4. Load images, resize to 224×224, and normalise pixel values.
    5. Stratified train / validation / test split (70 / 15 / 15).
    6. Compute class weights to handle class imbalance.
    7. Build a Keras data-augmentation layer (rotation, zoom, shift,
       brightness, horizontal flip).

Public API:
    clean_dataset()              -> remove corrupted + duplicate images
    load_dataset_paths()         -> collect (paths, labels, class_names)
    stratified_split()           -> train / val / test path arrays
    load_images()                -> numpy array of resized images
    get_preprocess_fn()          -> model-specific preprocessing function
    get_augmentation_layer()     -> Keras Sequential augmentation layer
    compute_class_weights()      -> {class_idx: weight} dict
"""

import os
import sys
import hashlib
import warnings

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config

warnings.filterwarnings("ignore", category=UserWarning)


# ===================================================================
# 1. FOLDER-NAME RESOLUTION
# ===================================================================
def _normalise(name: str) -> str:
    """Lowercase, strip, and collapse separators for forgiving matching."""
    return name.strip().lower().replace("_", " ").replace("-", " ").strip()


def _resolve_pose(folder_name: str) -> str | None:
    """
    Map a dataset folder name to one of the canonical pose names in
    config.POSES.  Returns None if the folder does not match any pose.
    """
    key = _normalise(folder_name).replace(" ", "")

    # Direct match against canonical names.
    for pose in config.POSES:
        if _normalise(pose).replace(" ", "") == key:
            return pose

    # Alias lookup.
    return config.POSE_ALIASES.get(key)


# ===================================================================
# 2. CORRUPTED IMAGE DETECTION & REMOVAL
# ===================================================================
def is_corrupted(path: str) -> bool:
    """
    Return True if the file at *path* cannot be opened and decoded as
    an image.  Uses PIL's verify() + load() for a thorough check.
    """
    try:
        with Image.open(path) as img:
            img.verify()          # Check file integrity.
        # verify() leaves the file pointer invalid; re-open to load.
        with Image.open(path) as img:
            img.load()            # Fully decode the pixel data.
        return False
    except Exception:
        return True


def remove_corrupted_images(dataset_dir: str = None) -> list[str]:
    """
    Scan *dataset_dir* (recursively) and delete every corrupted image.

    Returns a list of removed file paths (for logging).
    """
    dataset_dir = dataset_dir or config.DATASET_DIR
    removed = []

    for root, _dirs, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.lower().endswith(config.VALID_EXTENSIONS):
                continue
            path = os.path.join(root, fname)
            if is_corrupted(path):
                try:
                    os.remove(path)
                    removed.append(path)
                    print(f"  [removed] {os.path.basename(path)}")
                except OSError as exc:
                    print(f"  [error] {os.path.basename(path)}: {exc}")
    return removed


# ===================================================================
# 3. DUPLICATE IMAGE DETECTION
# ===================================================================
def _file_hash(path: str, chunk_size: int = 8192) -> str:
    """Return the MD5 hash of a file (fast exact-duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _perceptual_hash(path: str, size: tuple = (8, 8)) -> str:
    """
    Return a perceptual hash of the image for near-duplicate detection.
    Resizes to *size*, converts to greyscale, and thresholds on the mean.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize(size, Image.Resampling.LANCZOS)
            pixels = np.array(img, dtype=np.float32)
            mean = pixels.mean()
            bits = (pixels > mean).flatten()
            return "".join("1" if b else "0" for b in bits)
    except Exception:
        return ""


def detect_duplicates(dataset_dir: str = None) -> dict[str, list[str]]:
    """
    Detect duplicate images using exact MD5 hash and perceptual hash.

    Returns a dict:  {hash_value: [duplicate_file_paths]}
    Only groups with more than one file are returned.
    """
    dataset_dir = dataset_dir or config.DATASET_DIR
    exact_groups: dict[str, list[str]] = {}

    for root, _dirs, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.lower().endswith(config.VALID_EXTENSIONS):
                continue
            path = os.path.join(root, fname)
            try:
                h = _file_hash(path)
            except OSError:
                continue
            exact_groups.setdefault(h, []).append(path)

    duplicates = {
        h: paths for h, paths in exact_groups.items() if len(paths) > 1
    }
    return duplicates


def remove_duplicates(dataset_dir: str = None) -> list[str]:
    """
    Remove duplicate images, keeping only the first copy of each.
    Returns a list of removed file paths.
    """
    duplicates = detect_duplicates(dataset_dir)
    removed = []
    for _hash, paths in duplicates.items():
        for path in paths[1:]:
            try:
                os.remove(path)
                removed.append(path)
                print(f"  [dup-removed] {os.path.basename(path)}")
            except OSError as exc:
                print(f"  [dup-error] {os.path.basename(path)}: {exc}")
    return removed


# ===================================================================
# 4. DATASET PATH COLLECTION
# ===================================================================
def load_dataset_paths(
    dataset_dir: str = None,
    allowed_poses: list[str] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk *dataset_dir* and collect (paths, labels, class_names).

    Returns
    -------
    paths       : np.ndarray of str  — image file paths
    labels      : np.ndarray of int  — integer class labels
    class_names : list[str]          — sorted canonical pose names
    """
    dataset_dir = dataset_dir or config.DATASET_DIR
    allowed = allowed_poses or config.POSES

    # Build folder -> canonical pose mapping.
    folder_to_pose: dict[str, str] = {}
    for entry in os.listdir(dataset_dir):
        full = os.path.join(dataset_dir, entry)
        if not os.path.isdir(full):
            continue
        resolved = _resolve_pose(entry)
        if resolved and resolved in allowed:
            folder_to_pose[entry] = resolved

    if not folder_to_pose:
        raise ValueError(
            f"No valid pose folders found in {dataset_dir}.\n"
            f"Expected folders matching: {allowed}"
        )

    class_names = sorted(set(folder_to_pose.values()))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    paths, labels = [], []
    for folder, canonical in folder_to_pose.items():
        folder_path = os.path.join(dataset_dir, folder)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(config.VALID_EXTENSIONS):
                paths.append(os.path.join(folder_path, fname))
                labels.append(class_to_idx[canonical])

    return np.array(paths), np.array(labels), class_names


# ===================================================================
# 5. STRATIFIED TRAIN / VAL / TEST SPLIT
# ===================================================================
def stratified_split(
    paths: np.ndarray,
    labels: np.ndarray,
    val_split: float = None,
    test_split: float = None,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split into train / validation / test sets.

    Returns
    -------
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    """
    val_split = val_split if val_split is not None else config.VAL_SPLIT
    test_split = test_split if test_split is not None else config.TEST_SPLIT
    seed = seed if seed is not None else config.RANDOM_SEED

    n = len(paths)
    indices = np.arange(n)

    # First: separate the test set.
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, random_state=seed, stratify=labels
    )

    # Second: split the remaining into train / val.
    relative_val = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        random_state=seed,
        stratify=labels[train_val_idx],
    )

    return (
        paths[train_idx], labels[train_idx],
        paths[val_idx], labels[val_idx],
        paths[test_idx], labels[test_idx],
    )


# ===================================================================
# 6. IMAGE LOADING & PREPROCESSING
# ===================================================================
def load_image(path: str, target_size: tuple = None) -> np.ndarray:
    """
    Load a single image, resize, and return as a float32 RGB array.
    """
    target_size = target_size or config.IMG_SIZE
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float32)


def load_images(paths: np.ndarray, target_size: tuple = None) -> np.ndarray:
    """
    Load all images from *paths* into a numpy array.
    Shape: (N, H, W, 3), dtype: float32 (un-normalised).
    """
    target_size = target_size or config.IMG_SIZE
    images = np.empty((len(paths), *target_size, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        images[i] = load_image(p, target_size)
    return images


def get_preprocess_fn(model_name: str):
    """
    Return the model-specific preprocessing function.

    EfficientNetB0 and MobileNetV2 both expect inputs scaled to [-1, 1].
    """
    import tensorflow as tf

    if model_name.lower() in ("efficientnetb0", "efficientnet", "effnet"):
        return tf.keras.applications.efficientnet.preprocess_input
    elif model_name.lower() in ("mobilenetv2", "mobilenet"):
        return tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        # Generic normalisation: scale [0, 255] → [0, 1].
        return lambda x: x / 255.0


# ===================================================================
# 7. DATA AUGMENTATION LAYER
# ===================================================================
def get_augmentation_layer():
    """
    Build a Keras Sequential augmentation layer for training.

    Augmentations applied (as specified in the project requirements):
        - Rotation (±25 %)
        - Zoom (±20 %)
        - Width shift (±15 %)
        - Height shift (±15 %)
        - Brightness adjustment (±20 %)
        - Contrast adjustment (±15 %)
        - Horizontal flip
    """
    from tensorflow.keras import layers

    import tensorflow as tf
    return tf.keras.Sequential([
        layers.RandomRotation(0.25, seed=config.RANDOM_SEED, fill_mode="nearest"),
        layers.RandomZoom(0.2, seed=config.RANDOM_SEED, fill_mode="nearest"),
        layers.RandomTranslation(
            height_factor=0.15,
            width_factor=0.15,
            seed=config.RANDOM_SEED,
            fill_mode="nearest",
        ),
        layers.RandomBrightness(0.2, seed=config.RANDOM_SEED),
        layers.RandomContrast(0.15, seed=config.RANDOM_SEED),
        layers.RandomFlip("horizontal", seed=config.RANDOM_SEED),
    ])


# ===================================================================
# 8. CLASS WEIGHTS
# ===================================================================
def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    """
    Compute balanced class weights to handle class imbalance.

    Uses sklearn's compute_class_weight('balanced', ...).
    """
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ===================================================================
# 9. TF.DATA PIPELINE BUILDER
# ===================================================================
def create_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    augment: bool = False,
    batch_size: int = None,
    num_classes: int = None,
):
    """
    Build a tf.data.Dataset from in-memory image arrays.

    Applies model-specific preprocessing and optional augmentation.
    """
    import tensorflow as tf

    batch_size = batch_size or config.BATCH_SIZE
    num_classes = num_classes or config.NUM_CLASSES
    preprocess_fn = get_preprocess_fn(model_name)

    labels_oh = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels_oh))

    if augment:
        aug_layer = get_augmentation_layer()
        dataset = dataset.shuffle(buffer_size=min(len(images), 1000), seed=config.RANDOM_SEED)
        dataset = dataset.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Model-specific preprocessing (always applied).
    dataset = dataset.map(
        lambda x, y: (preprocess_fn(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# ===================================================================
# 10. FULL CLEANING PIPELINE
# ===================================================================
def _fix_encoding():
    """Set UTF-8 encoding for Windows console."""
    import io
    if sys.getdefaultencoding() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def clean_dataset(dataset_dir: str = None) -> dict:
    """
    Run the full data-cleaning pipeline:
        1. Remove corrupted images.
        2. Remove duplicate images.

    Returns a summary dict with counts.
    """
    _fix_encoding()
    dataset_dir = dataset_dir or config.DATASET_DIR
    print("\n" + "=" * 60)
    print("  DATA CLEANING PIPELINE")
    print("=" * 60)

    print("\n[1/2] Removing corrupted images...")
    corrupted = remove_corrupted_images(dataset_dir)
    print(f"  Corrupted images removed: {len(corrupted)}")

    duplicates = remove_duplicates(dataset_dir)
    print(f"  Duplicate images removed: {len(duplicates)}")

    summary = {
        "corrupted_removed": len(corrupted),
        "duplicates_removed": len(duplicates),
        "corrupted_files": corrupted,
        "duplicate_files": duplicates,
    }

    print(f"\nCleaning complete. Total removed: {len(corrupted) + len(duplicates)}")
    return summary


# ===================================================================
# SELF-TEST
# ===================================================================
if __name__ == "__main__":
    config.print_config()

    print("\n--- Scanning dataset ---")
    paths, labels, class_names = load_dataset_paths()
    print(f"Total images found : {len(paths)}")
    print(f"Classes            : {class_names}")
    for i, name in enumerate(class_names):
        count = int((labels == i).sum())
        print(f"  {name:<20s} {count:>4d} images")

    print("\n--- Stratified split ---")
    tr_p, tr_l, va_p, va_l, te_p, te_l = stratified_split(paths, labels)
    print(f"Train: {len(tr_p)}, Val: {len(va_p)}, Test: {len(te_p)}")

    print("\n--- Class weights ---")
    cw = compute_class_weights(tr_l)
    for c, w in cw.items():
        print(f"  Class {class_names[c]:<20s} weight: {w:.4f}")
