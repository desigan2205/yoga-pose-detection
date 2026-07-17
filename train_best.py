import os, sys, json, gc, warnings, itertools, shutil, time
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, mixed_precision
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Loss, CategoricalCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config
from data_processing import load_dataset_paths, load_images, load_image

print(f"TensorFlow: {tf.__version__}, Keras: {tf.keras.__version__}")
print(f"GPU Available: {bool(tf.config.list_physical_devices('GPU'))}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    mixed_precision.set_global_policy('mixed_float16')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

OUTPUT_DIR = config.OUTPUT_DIR
MODEL_DIR = config.MODEL_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LABELS_PATH = config.LABELS_PATH

BS = 32
EPOCHS_P1 = 60
EPOCHS_P2 = 30
LR_P1 = 5e-4
LR_P2 = 1e-5
LABEL_SMOOTH = 0.15
DROPOUT_RATE = 0.4
L2_FACTOR = 5e-5
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.75, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.ce = CategoricalCrossentropy(from_logits=from_logits, label_smoothing=LABEL_SMOOTH)

    def call(self, y_true, y_pred):
        ce_loss = self.ce(y_true, y_pred)
        p_t = tf.exp(-ce_loss)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        alpha_factor = tf.ones_like(ce_loss) * self.alpha
        alpha_t = tf.where(tf.reduce_max(y_true, axis=-1) > 0.5,
                           alpha_factor, 1.0 - alpha_factor)
        return alpha_t * focal_weight * ce_loss

    def get_config(self):
        d = super().get_config()
        d.update({"gamma": self.gamma, "alpha": self.alpha, "from_logits": self.from_logits})
        return d


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------
def build_model(arch_name, num_classes):
    base_map = {
        "efficientnetv2b0": (EfficientNetV2B0, tf.keras.applications.efficientnet_v2.preprocess_input),
        "efficientnetb0": (EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
        "mobilenetv2": (MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
        "resnet50": (ResNet50, tf.keras.applications.resnet50.preprocess_input),
    }
    base_cls, preprocess_fn = base_map[arch_name]
    base = base_cls(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(L2_FACTOR))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(L2_FACTOR))(x)
    x = layers.Dropout(DROPOUT_RATE * 0.75)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(
        num_classes, activation="softmax",
        kernel_regularizer=regularizers.l2(L2_FACTOR),
    )(x)

    model = models.Model(inputs=base.input, outputs=outputs, name=arch_name)
    return model, base, preprocess_fn


def get_preprocess_fn(arch_name):
    base_map = {
        "efficientnetv2b0": tf.keras.applications.efficientnet_v2.preprocess_input,
        "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
        "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "resnet50": tf.keras.applications.resnet50.preprocess_input,
    }
    return base_map[arch_name]


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------
def mixup_batch(images, labels_onehot, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform((), 0.0, 1.0, dtype=tf.float32)
    lam = tf.maximum(lam, 1.0 - lam)
    lam = lam * alpha + (1.0 - alpha) * 0.5
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = tf.cast(lam, images.dtype) * images + tf.cast(1.0 - lam, images.dtype) * tf.gather(images, indices)
    mixed_labels = lam * labels_onehot + (1.0 - lam) * tf.gather(labels_onehot, indices)
    return mixed_images, mixed_labels


def get_augmentation():
    return tf.keras.Sequential([
        layers.RandomRotation(0.3, seed=RANDOM_SEED, fill_mode="nearest"),
        layers.RandomZoom(0.25, seed=RANDOM_SEED, fill_mode="nearest"),
        layers.RandomTranslation(0.15, 0.15, seed=RANDOM_SEED, fill_mode="nearest"),
        layers.RandomBrightness(0.25, seed=RANDOM_SEED),
        layers.RandomContrast(0.2, seed=RANDOM_SEED),
        layers.RandomFlip("horizontal", seed=RANDOM_SEED),
    ], name="augmentation_layer")


def build_tf_dataset(images, labels, preprocess_fn, num_classes,
                     augment=False, mixup=False, batch_size=BS):
    labels_oh = tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((images, labels_oh))

    if augment:
        aug_layer = get_augmentation()
        ds = ds.shuffle(buffer_size=min(len(images), 3000), seed=RANDOM_SEED)
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.map(
        lambda x, y: (preprocess_fn(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if mixup and augment:
        ds = ds.map(
            lambda x, y: mixup_batch(x, y, alpha=0.2),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds


def stratified_split(paths, labels):
    indices = np.arange(len(paths))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )
    relative_val = VAL_SPLIT / (1.0 - TEST_SPLIT)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val,
        random_state=RANDOM_SEED, stratify=labels[train_val_idx]
    )
    return (paths[train_idx], labels[train_idx],
            paths[val_idx], labels[val_idx],
            paths[test_idx], labels[test_idx])


def compute_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    weights = np.power(weights, 0.7)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_phase(model, train_ds, val_ds, class_weights_dict, save_path,
                epochs, lr, phase_name, patience_es=12, patience_lr=4):
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    optimizer = AdamW(learning_rate=lr, weight_decay=1e-5)
    model.compile(
        optimizer=optimizer,
        loss=FocalLoss(gamma=2.0, alpha=0.75),
        metrics=["accuracy"],
    )

    callbacks = [
        ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True,
                        mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience_es,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience_lr,
                          min_lr=1e-7, verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights_dict if "Phase 1" in phase_name else None,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, test_images, test_labels, preprocess_fn, class_names, model_name):
    test_pp = preprocess_fn(test_images)
    test_oh = tf.keras.utils.to_categorical(test_labels, len(class_names))

    loss, acc = model.evaluate(test_pp, test_oh, verbose=0)
    y_prob = model.predict(test_pp, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_labels

    present_labels = sorted(set(y_true) | set(y_pred))
    present_names = [class_names[i] for i in present_labels]
    results = {
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=present_labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=present_labels,
            target_names=present_names, digits=4, output_dict=True, zero_division=0,
        ),
        "class_names": class_names,
    }

    print(f"\n{'='*60}")
    print(f"  {model_name} — TEST EVALUATION")
    print(f"{'='*60}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision (w) : {results['precision']:.4f}")
    print(f"  Recall    (w) : {results['recall']:.4f}")
    print(f"  F1 Score  (w) : {results['f1_score']:.4f}")

    return results, y_pred, y_true


# ---------------------------------------------------------------------------
# Plotting / saving
# ---------------------------------------------------------------------------
def save_plots(results, history, class_names, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(results["confusion_matrix"])
    # If CM is smaller than class_names, only some labels were present
    cm_labels = class_names[:cm.shape[0]] if cm.shape[0] <= len(class_names) else class_names

    # Confusion Matrix
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar(shrink=0.8)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, cm_labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(tick_marks, cm_labels, fontsize=8)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {cm_path}")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["accuracy"], label="Train", linewidth=1.5)
    ax1.plot(history["val_accuracy"], label="Validation", linewidth=1.5)
    ax1.set_title(f"{model_name} — Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.plot(history["loss"], label="Train", linewidth=1.5)
    ax2.plot(history["val_loss"], label="Validation", linewidth=1.5)
    ax2.set_title(f"{model_name} — Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Yoga Pose Detection — Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, f"training_history_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {hist_path}")


def save_classification_report_text(report_dict, class_names, save_path):
    lines = [
        "=" * 70,
        "  YOGA POSE DETECTION — CLASSIFICATION REPORT",
        "=" * 70,
        "",
        f"{'Class':<25s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}",
        "-" * 65,
    ]
    for cls in class_names:
        if cls in report_dict:
            m = report_dict[cls]
            lines.append(
                f"{cls:<25s} {m.get('precision', 0):>10.4f} {m.get('recall', 0):>10.4f} "
                f"{m.get('f1-score', 0):>10.4f} {m.get('support', 0):>10.0f}"
            )
        else:
            lines.append(f"{cls:<25s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {'0':>10s}")
    lines.append("-" * 65)
    if "weighted avg" in report_dict:
        w = report_dict["weighted avg"]
        lines.append(
            f"{'Weighted Avg':<25s} {w.get('precision', 0):>10.4f} {w.get('recall', 0):>10.4f} "
            f"{w.get('f1-score', 0):>10.4f} {w.get('support', 0):>10.0f}"
        )
    if "accuracy" in report_dict:
        lines.append(
            f"{'Accuracy':<25s} {'':>10s} {'':>10s} {report_dict['accuracy']:>10.4f} {'':>10s}"
        )
    lines.extend(["", "=" * 70])

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {save_path}")


def save_labels(class_names):
    mapping = {str(i): name for i, name in enumerate(class_names)}
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"  [SAVED] {LABELS_PATH}")


# ---------------------------------------------------------------------------
# Main training + comparison
# ---------------------------------------------------------------------------
def train():
    t0 = time.time()
    print("=" * 60)
    print("  YOGA POSE DETECTION — ARCHITECTURE COMPARISON & TRAINING")
    print("=" * 60)
    config.print_config()

    # Verify dataset path exists
    ds_path = os.path.abspath(config.DATASET_DIR)
    if not os.path.isdir(ds_path):
        print(f"\n  [ERROR] Dataset directory not found: {ds_path}")
        print(f"  Current working dir: {os.getcwd()}")
        print(f"  Contents of parent: {os.listdir(os.path.dirname(ds_path)) if os.path.isdir(os.path.dirname(ds_path)) else 'N/A'}")
        print(f"  BASE_DIR in config: {config.BASE_DIR}")
        raise FileNotFoundError(f"Dataset directory does not exist: {ds_path}")

    print(f"\n{'='*60}")
    print("  STEP 0: Cleaning dataset (corrupted + duplicate removal)")
    print(f"{'='*60}")
    from data_processing import clean_dataset
    try:
        clean_summary = clean_dataset()
    except Exception as e:
        print(f"  Clean skipped (dataset may already be clean): {e}")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  [1/7] Loading dataset...")
    print(f"{'='*60}")
    paths, labels, class_names = load_dataset_paths()
    num_classes = len(class_names)
    print(f"  Total images: {len(paths)} across {num_classes} classes")
    for i, name in enumerate(class_names):
        count = int((labels == i).sum())
        print(f"    {name:<25s} {count:>4d} images")

    # ------------------------------------------------------------------
    # 2. Stratified split
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  [2/7] Stratified train/val/test split...")
    print(f"{'='*60}")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = stratified_split(
        paths, labels
    )
    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # ------------------------------------------------------------------
    # 3. Class weights
    # ------------------------------------------------------------------
    class_weights = compute_class_weights(train_labels)
    print(f"\n{'='*60}")
    print("  [3/7] Class weights computed")
    print(f"{'='*60}")
    for k, v in class_weights.items():
        print(f"    {class_names[k]:<25s} weight={v:.4f}")

    # ------------------------------------------------------------------
    # 4. Load images into memory
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  [4/7] Loading & preprocessing images...")
    print(f"{'='*60}")
    train_images = load_images(train_paths)
    val_images = load_images(val_paths)
    test_images = load_images(test_paths)
    print(f"  Train: {train_images.shape}")
    print(f"  Val:   {val_images.shape}")
    print(f"  Test:  {test_images.shape}")

    save_labels(class_names)

    # ------------------------------------------------------------------
    # 5. Architecture comparison
    # ------------------------------------------------------------------
    ARCHITECTURES = [
        "efficientnetv2b0",
        "efficientnetb0",
        "mobilenetv2",
        "resnet50",
    ]

    all_results = {}
    best_overall_val_acc = 0
    best_overall_name = ""
    best_overall_model_path = ""

    print(f"\n{'='*60}")
    print("  [5/7] Training & comparing architectures")
    print(f"{'='*60}")

    for arch_name in ARCHITECTURES:
        save_path = os.path.join(MODEL_DIR, f"best_{arch_name}.keras")
        done_marker = save_path + ".done"

        # Resume: skip if fully trained
        if os.path.exists(done_marker):
            print(f"\n  [{arch_name.upper()}] already fully trained. Skipping to evaluation.")
            model = tf.keras.models.load_model(
                save_path, custom_objects={"FocalLoss": FocalLoss},
            )
            preprocess_fn = get_preprocess_fn(arch_name)
            results, y_pred, y_true = evaluate_model(
                model, test_images, test_labels, preprocess_fn, class_names, arch_name.upper()
            )
            save_plots(results, {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}, class_names, arch_name.upper())
            per_model_cr_path = os.path.join(
                OUTPUT_DIR, f"classification_report_{arch_name}.txt"
            )
            save_classification_report_text(
                results["classification_report"], class_names, per_model_cr_path
            )
            test_acc = results["accuracy"]
            all_results[arch_name] = {
                "results": results,
                "history": {},
                "val_accuracy": 0.0,
                "test_accuracy": test_acc,
                "f1_score": results["f1_score"],
                "save_path": save_path,
            }
            gc.collect()
            if test_acc > best_overall_val_acc:
                best_overall_val_acc = test_acc
                best_overall_name = arch_name
                best_overall_model_path = save_path
            continue

        # Resume: if Phase 1 checkpoint exists but no .done marker, skip Phase 1
        phase1_done = os.path.exists(save_path) and not os.path.exists(done_marker)

        print(f"\n{'='*60}")
        print(f"  >>> ARCHITECTURE: {arch_name.upper()}")
        print(f"{'='*60}")

        model, base_model, preprocess_fn = build_model(arch_name, num_classes)

        if not phase1_done:
            # Fresh start – run Phase 1
            model.summary()
            train_ds_p1 = build_tf_dataset(
                train_images, train_labels, preprocess_fn, num_classes,
                augment=True, mixup=True,
            )
            val_ds_p1 = build_tf_dataset(
                val_images, val_labels, preprocess_fn, num_classes,
                augment=False, mixup=False,
            )
            history_p1 = train_phase(
                model, train_ds_p1, val_ds_p1, class_weights, save_path,
                EPOCHS_P1, LR_P1,
                f"PHASE 1: {arch_name} — Frozen Base + Focal Loss + MixUp",
                patience_es=15, patience_lr=5,
            )
        else:
            # Resume from Phase 1 checkpoint – skip to Phase 2
            print(f"\n  [{arch_name.upper()}] Phase 1 checkpoint found. Skipping to Phase 2.")
            model = tf.keras.models.load_model(
                save_path, custom_objects={"FocalLoss": FocalLoss},
            )
            history_p1 = tf.keras.callbacks.History()
            history_p1.history = {}  # placeholder, will be overwritten on merge if Phase 2 runs

        # Phase 2: Fine-tuning
        print(f"\n{'='*60}")
        print(f"  PHASE 2: {arch_name} — Fine-tuning")
        print(f"{'='*60}")

        model = tf.keras.models.load_model(
            save_path, custom_objects={"FocalLoss": FocalLoss},
        )

        for layer in model.layers:
            layer.trainable = True

        total_layers = len(model.layers)
        unfreeze_from = int(total_layers * 0.45)
        for layer in model.layers[:unfreeze_from]:
            if not isinstance(layer, layers.Dropout):
                layer.trainable = False

        trainable_count = sum(1 for l in model.layers if l.trainable)
        print(f"  Unfrozen: {trainable_count}/{total_layers} layers (from layer {unfreeze_from})")

        train_ds_p2 = build_tf_dataset(
            train_images, train_labels, preprocess_fn, num_classes,
            augment=True, mixup=True,
        )
        val_ds_p2 = build_tf_dataset(
            val_images, val_labels, preprocess_fn, num_classes,
            augment=False, mixup=False,
        )

        history_p2 = train_phase(
            model, train_ds_p2, val_ds_p2, class_weights, save_path,
            EPOCHS_P2, LR_P2,
            f"PHASE 2: {arch_name} — Fine-tuning",
            patience_es=8, patience_lr=3,
        )

        # Mark as fully trained
        with open(done_marker, "w") as f:
            f.write(f"{arch_name} training completed at {datetime.now()}\n")

        # Merge histories
        combined_history = {}
        for key in history_p1.history:
            combined_history[key] = (
                history_p1.history[key] + history_p2.history.get(key, [])
            )
        for key in history_p2.history:
            if key not in combined_history:
                combined_history[key] = history_p2.history[key]

        # Evaluate
        print(f"\n{'='*60}")
        print(f"  Evaluating {arch_name} on test set...")
        print(f"{'='*60}")
        if os.path.exists(save_path):
            model = tf.keras.models.load_model(
                save_path,
                custom_objects={"FocalLoss": FocalLoss},
            )

        results, y_pred, y_true = evaluate_model(
            model, test_images, test_labels, preprocess_fn, class_names, arch_name.upper()
        )

        # Save per-architecture outputs
        save_plots(results, combined_history, class_names, arch_name.upper())

        per_model_cr_path = os.path.join(
            OUTPUT_DIR, f"classification_report_{arch_name}.txt"
        )
        save_classification_report_text(
            results["classification_report"], class_names, per_model_cr_path
        )

        all_results[arch_name] = {
            "results": results,
            "history": combined_history,
            "val_accuracy": max(combined_history.get("val_accuracy", [0])),
            "test_accuracy": results["accuracy"],
            "f1_score": results["f1_score"],
            "save_path": save_path,
        }

        val_acc = max(combined_history.get("val_accuracy", [0]))
        print(f"\n  >>> {arch_name}: Best Val Acc={val_acc:.4f}, Test Acc={results['accuracy']:.4f}")

        if val_acc > best_overall_val_acc:
            best_overall_val_acc = val_acc
            best_overall_name = arch_name
            best_overall_model_path = save_path

        gc.collect()
        if gpus:
            tf.keras.backend.clear_session()

    # ------------------------------------------------------------------
    # 6. Select & save best model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  [6/7] Selecting best model")
    print(f"{'='*60}")

    print(f"\n  Architecture Comparison Results:")
    print(f"  {'='*50}")
    print(f"  {'Architecture':<25s} {'Val Acc':>10s} {'Test Acc':>10s} {'F1 Score':>10s}")
    print(f"  {'-'*55}")
    for arch in ARCHITECTURES:
        info = all_results[arch]
        print(f"  {arch:<25s} {info['val_accuracy']:>8.4f}  {info['test_accuracy']:>8.4f}  {info['f1_score']:>8.4f}")

    print(f"\n  >>> BEST ARCHITECTURE: {best_overall_name} (val_acc={best_overall_val_acc:.4f})")

    # Copy best model to main model path
    if os.path.exists(best_overall_model_path):
        shutil.copy2(best_overall_model_path, config.BEST_MODEL_PATH)
        shutil.copy2(best_overall_model_path, config.MODEL_PATH)
        print(f"\n  Copied best model to:")
        print(f"    -> {config.BEST_MODEL_PATH}")
        print(f"    -> {config.MODEL_PATH}")

    # ------------------------------------------------------------------
    # 7. Generate final evaluation outputs
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  [7/7] Generating final evaluation outputs")
    print(f"{'='*60}")

    best_info = all_results[best_overall_name]
    best_results = best_info["results"]
    best_history = best_info["history"]

    # Load best model for final outputs
    best_model = tf.keras.models.load_model(
        config.BEST_MODEL_PATH,
        custom_objects={"FocalLoss": FocalLoss},
    )

    # Confusion matrix (best)
    cm = np.array(best_results["confusion_matrix"])
    plt_confusion_matrix_final(cm, class_names, best_overall_name)

    # Training curves (best)
    plt_training_history_final(best_history, best_overall_name)

    # Classification report (best)
    save_classification_report_text(
        best_results["classification_report"], class_names,
        config.CLASSIFICATION_REPORT_PATH,
    )

    # Evaluation metrics JSON
    with open(config.EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        report_data = {
            "best_architecture": best_overall_name,
            "best_val_accuracy": float(best_overall_val_acc),
            "test_accuracy": float(best_results["accuracy"]),
            "test_precision": float(best_results["precision"]),
            "test_recall": float(best_results["recall"]),
            "test_f1_score": float(best_results["f1_score"]),
            "test_loss": float(best_results["test_loss"]),
            "class_names": class_names,
            "architecture_comparison": {
                arch: {
                    "val_accuracy": float(all_results[arch]["val_accuracy"]),
                    "test_accuracy": float(all_results[arch]["test_accuracy"]),
                    "f1_score": float(all_results[arch]["f1_score"]),
                }
                for arch in ARCHITECTURES
            },
            "confusion_matrix": best_results["confusion_matrix"],
            "timestamp": datetime.now().isoformat(),
        }
        json.dump(report_data, f, indent=2)
    print(f"  [SAVED] {config.EVALUATION_METRICS_PATH}")

    # Save best training history
    history_serializable = {
        k: [float(v) for v in vals]
        for k, vals in best_history.items() if isinstance(vals, list)
    }
    with open(config.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_serializable, f, indent=2)
    print(f"  [SAVED] {config.HISTORY_PATH}")

    # Performance summary
    duration = time.time() - t0
    summary_path = os.path.join(OUTPUT_DIR, "performance_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  YOGA POSE DETECTION — PERFORMANCE SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training duration: {duration:.0f}s ({duration/60:.1f} min)\n")
        f.write(f"Dataset size: {len(paths)} images\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"GPU available: {bool(gpus)}\n")
        if gpus:
            f.write(f"Mixed precision: mixed_float16\n")
        f.write(f"\nArchitecture comparison (best → worst):\n")
        sorted_archs = sorted(
            ARCHITECTURES,
            key=lambda a: all_results[a]["val_accuracy"],
            reverse=True,
        )
        f.write(f"  {'Rank':<6s} {'Architecture':<25s} {'Val Acc':>10s} {'Test Acc':>10s} {'F1':>10s}\n")
        f.write(f"  {'-'*61}\n")
        for rank, arch in enumerate(sorted_archs, 1):
            info = all_results[arch]
            f.write(f"  {rank:<6d} {arch:<25s} {info['val_accuracy']:>8.4f}  {info['test_accuracy']:>8.4f}  {info['f1_score']:>8.4f}\n")
        f.write(f"\nBest architecture: {best_overall_name}\n")
        f.write(f"Best validation accuracy: {best_overall_val_acc:.4f} ({best_overall_val_acc*100:.2f}%)\n")
        f.write(f"Test accuracy: {best_results['accuracy']:.4f} ({best_results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision (weighted): {best_results['precision']:.4f}\n")
        f.write(f"Recall (weighted): {best_results['recall']:.4f}\n")
        f.write(f"F1 Score (weighted): {best_results['f1_score']:.4f}\n")
        f.write(f"Test loss: {best_results['test_loss']:.4f}\n")

        # Per-class breakdown
        f.write(f"\n{'='*70}\n")
        f.write("  PER-CLASS BREAKDOWN (Best Model)\n")
        f.write(f"{'='*70}\n")
        cr = best_results["classification_report"]
        f.write(f"\n{'Class':<25s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}\n")
        f.write(f"{'-'*65}\n")
        for cls in class_names:
            if cls in cr:
                m = cr[cls]
                f.write(
                    f"{cls:<25s} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                    f"{m['f1-score']:>10.4f} {m['support']:>10.0f}\n"
                )
        f.write(f"\nConfusion matrix saved to: {os.path.join(OUTPUT_DIR, 'confusion_matrix_best.png')}\n")
        f.write(f"Training plots saved to: {os.path.join(OUTPUT_DIR, 'training_history_best.png')}\n")
        f.write(f"Model saved to: {config.BEST_MODEL_PATH}\n")
    print(f"  [SAVED] {summary_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Duration        : {duration:.0f}s ({duration/60:.1f} min)")
    print(f"  Best architecture: {best_overall_name}")
    print(f"  Best val acc    : {best_overall_val_acc:.4f} ({best_overall_val_acc*100:.2f}%)")
    print(f"  Test accuracy   : {best_results['accuracy']:.4f} ({best_results['accuracy']*100:.2f}%)")
    print(f"  F1 Score        : {best_results['f1_score']:.4f}")
    print(f"\n  Outputs:")
    print(f"    Best model        : {config.BEST_MODEL_PATH}")
    print(f"    Primary model     : {config.MODEL_PATH}")
    print(f"    Labels            : {LABELS_PATH}")
    print(f"    Confusion matrix  : {config.CONFUSION_MATRIX_PATH}")
    print(f"    Training plot     : {config.TRAINING_HISTORY_PATH}")
    print(f"    Classification rpt: {config.CLASSIFICATION_REPORT_PATH}")
    print(f"    Metrics JSON      : {config.EVALUATION_METRICS_PATH}")
    print(f"    Performance summ  : {summary_path}")
    print(f"{'='*60}")

    return best_results


def plt_confusion_matrix_final(cm, class_names, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{model_name.upper()} — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar(shrink=0.8)
    cm_labels = class_names[:cm.shape[0]]
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, cm_labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(tick_marks, cm_labels, fontsize=8)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {config.CONFUSION_MATRIX_PATH}")


def plt_training_history_final(history, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["accuracy"], label="Train", linewidth=1.5)
    ax1.plot(history["val_accuracy"], label="Validation", linewidth=1.5)
    ax1.set_title(f"{model_name.upper()} — Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.plot(history["loss"], label="Train", linewidth=1.5)
    ax2.plot(history["val_loss"], label="Validation", linewidth=1.5)
    ax2.set_title(f"{model_name.upper()} — Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Yoga Pose Detection — Training History (Best Model)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.TRAINING_HISTORY_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {config.TRAINING_HISTORY_PATH}")


if __name__ == "__main__":
    results = train()
