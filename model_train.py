"""
model_train.py
==============
Train yoga pose classifiers with transfer learning.

Architectures (in order of preference):
    1. EfficientNetB0 (primary) — best accuracy-size trade-off
    2. MobileNetV2 (backup)     — faster, lighter

Two-phase training:
    Phase 1 — Frozen base: train only the classification head (100 epochs max)
    Phase 2 — Fine-tuning:  unfreeze top layers of base, very low LR (50 epochs max)

Outputs (saved to config.MODEL_DIR and config.OUTPUT_DIR):
    - best_model.keras              (EfficientNetB0)
    - best_model_mobilenetv2.keras  (MobileNetV2)
    - class_labels.json
    - training_history.json
    - confusion_matrix.png
    - training_accuracy.png
    - classification_report.txt
    - evaluation_metrics.json

Usage:
    python model_train.py                     # train both models
    python model_train.py --model efficient   # train only EfficientNetB0
    python model_train.py --model mobilenet   # train only MobileNetV2
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

import config
from data_processing import (
    load_dataset_paths, stratified_split, load_images,
    get_preprocess_fn, get_augmentation_layer, compute_class_weights,
)

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")


# ===================================================================
# 1. MODEL BUILDERS
# ===================================================================

def build_efficientnetb0(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build EfficientNetB0 with frozen base and custom classification head."""
    base = EfficientNetB0(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.Dropout(config.DROPOUT_1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.Dropout(config.DROPOUT_2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    outputs = layers.Dense(
        num_classes, activation="softmax",
        kernel_regularizer=regularizers.l2(config.L2_REG),
    )(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    return model, base


def build_mobilenetv2(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build MobileNetV2 with frozen base and custom classification head."""
    base = MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.Dropout(config.DROPOUT_1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.Dropout(config.DROPOUT_2)(x)
    outputs = layers.Dense(
        num_classes, activation="softmax",
        kernel_regularizer=regularizers.l2(config.L2_REG),
    )(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    return model, base


MODEL_REGISTRY = {
    "efficientnetb0": {
        "build_fn": build_efficientnetb0,
        "preprocess_fn": tf.keras.applications.efficientnet.preprocess_input,
        "save_path": config.MODEL_PATH,
    },
    "mobilenetv2": {
        "build_fn": build_mobilenetv2,
        "preprocess_fn": tf.keras.applications.mobilenet_v2.preprocess_input,
        "save_path": config.MODEL_BACKUP_PATH,
    },
}


# ===================================================================
# 2. DATA PIPELINE
# ===================================================================

def build_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    preprocess_fn,
    num_classes: int,
    augment: bool = False,
    batch_size: int = None,
):
    """Build tf.data.Dataset with optional augmentation."""
    batch_size = batch_size or config.BATCH_SIZE
    labels_oh = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    ds = tf.data.Dataset.from_tensor_slices((images, labels_oh))

    if augment:
        aug_layer = get_augmentation_layer()
        ds = ds.shuffle(buffer_size=min(len(images), 2000), seed=config.RANDOM_SEED)
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.map(
        lambda x, y: (preprocess_fn(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ===================================================================
# 3. TRAINING LOOP
# ===================================================================

def train_model(
    model_name: str,
    train_ds,
    val_ds,
    class_weights: dict,
    num_classes: int,
    save_path: str,
):
    """Run two-phase training for a given model."""
    print(f"\n{'='*60}")
    print(f"  TRAINING {model_name.upper()}")
    print(f"{'='*60}")

    # --- Phase 1: Frozen base ---
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Frozen base — training classification head")
    print(f"{'='*60}")

    model_tuple = MODEL_REGISTRY[model_name]
    model, base_model = model_tuple["build_fn"](num_classes)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    callbacks_p1 = [
        ModelCheckpoint(
            save_path, monitor="val_accuracy", save_best_only=True,
            mode="max", verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE, min_lr=config.REDUCE_LR_MIN,
            verbose=1,
        ),
    ]

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # --- Phase 2: Fine-tuning ---
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Fine-tuning top layers")
    print(f"{'='*60}")

    # Reload best weights from phase 1
    if os.path.exists(save_path):
        model = tf.keras.models.load_model(save_path)

    # Identify base (non-head) layers by name exclusion
    head_names = {'global_average_pooling2d', 'batch_normalization', 'batch_normalization_1',
                  'batch_normalization_2', 'dropout', 'dropout_1', 'dense', 'dense_1', 'dense_2', 'dense_3'}
    base_layers = [l for l in model.layers if l.name not in head_names]

    # Unfreeze base layers from FINE_TUNE_UNFREEZE_FROM onwards
    for layer in base_layers:
        layer.trainable = True
    for layer in base_layers[:config.FINE_TUNE_UNFREEZE_FROM]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_layers if l.trainable)
    print(f"  Unfrozen layers: {trainable_count} / {len(base_layers)}")

    model.compile(
        optimizer=Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
        loss=CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    callbacks_p2 = [
        ModelCheckpoint(
            save_path, monitor="val_accuracy", save_best_only=True,
            mode="max", verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=max(5, config.EARLY_STOP_PATIENCE // 2),
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE // 2, min_lr=config.REDUCE_LR_MIN,
            verbose=1,
        ),
    ]

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_p2,
        verbose=1,
    )

    # Combine histories
    combined = {}
    for key in history_p1.history:
        combined[key] = history_p1.history[key] + history_p2.history.get(key, [])
    for key in history_p2.history:
        if key not in combined:
            combined[key] = history_p2.history[key]

    return model, combined, history_p1, history_p2


# ===================================================================
# 4. EVALUATION
# ===================================================================

def evaluate_on_test(
    model, test_images: np.ndarray, test_labels: np.ndarray,
    preprocess_fn, class_names: list[str],
) -> dict:
    """Run comprehensive evaluation on held-out test set."""
    test_pp = preprocess_fn(test_images)
    test_oh = tf.keras.utils.to_categorical(test_labels, len(class_names))

    loss, acc = model.evaluate(test_pp, test_oh, verbose=0)
    y_prob = model.predict(test_pp, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_labels

    results = {
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4, output_dict=True,
        ),
        "class_names": class_names,
    }

    print(f"\n  Test loss      : {results['test_loss']:.4f}")
    print(f"  Test accuracy  : {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"  Precision      : {results['precision']:.4f}")
    print(f"  Recall         : {results['recall']:.4f}")
    print(f"  F1 Score       : {results['f1_score']:.4f}")

    return results, y_pred, y_true


# ===================================================================
# 5. SAVE OUTPUTS
# ===================================================================

def save_plots(history: dict, model_name: str):
    """Save training accuracy and loss plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    plt.savefig(config.TRAINING_HISTORY_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training plot -> {config.TRAINING_HISTORY_PATH}")


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str = "Confusion Matrix"):
    """Save confusion matrix plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import itertools

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.colorbar(shrink=0.8)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=11,
        )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix -> {config.CONFUSION_MATRIX_PATH}")


def save_classification_report(report_dict: dict, class_names: list[str]):
    """Save classification report as formatted text file."""
    lines = []
    lines.append("=" * 70)
    lines.append("  YOGA POSE DETECTION — CLASSIFICATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    lines.append("-" * 60)

    for cls in class_names:
        if cls in report_dict:
            metrics = report_dict[cls]
            lines.append(
                f"{cls:<20s} {metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} {metrics['f1-score']:>10.4f} "
                f"{metrics['support']:>10.0f}"
            )

    lines.append("-" * 60)
    if "weighted avg" in report_dict:
        w = report_dict["weighted avg"]
        lines.append(
            f"{'Weighted Avg':<20s} {w['precision']:>10.4f} "
            f"{w['recall']:>10.4f} {w['f1-score']:>10.4f} "
            f"{w['support']:>10.0f}"
        )
    if "accuracy" in report_dict:
        acc = report_dict["accuracy"]
        lines.append(
            f"{'Accuracy':<20s} {'':>10s} {'':>10s} "
            f"{acc:>10.4f} {'':>10s}"
        )

    lines.append("")
    lines.append("=" * 70)

    with open(config.CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved classification report -> {config.CLASSIFICATION_REPORT_PATH}")


def save_labels(class_names: list[str]):
    """Save class label mapping to JSON."""
    mapping = {str(i): name for i, name in enumerate(class_names)}
    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Saved class labels -> {config.LABELS_PATH}")


def save_training_history(history: dict, model_name: str):
    """Save training history to JSON."""
    serializable = {}
    for key, val in history.items():
        if isinstance(val, list):
            serializable[key] = [float(v) for v in val]
    serializable["model_name"] = model_name
    with open(config.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_evaluation_metrics(
    results_primary: dict, results_backup: dict = None,
):
    """Save evaluation metrics for all models to JSON."""
    all_metrics = {"primary": results_primary}
    if results_backup:
        all_metrics["backup"] = results_backup
    with open(config.EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved evaluation metrics -> {config.EVALUATION_METRICS_PATH}")


# ===================================================================
# 6. MAIN TRAINING PIPELINE
# ===================================================================

def train(selected_models: list[str] = None, skip_finetune: bool = False):
    """
    Run the full training pipeline for selected models.

    Parameters
    ----------
    selected_models : list of str, optional
        Which models to train: 'efficientnetb0', 'mobilenetv2', or both.
        Defaults to both.
    skip_finetune : bool
        If True, skip phase 2 fine-tuning.
    """
    selected_models = selected_models or list(MODEL_REGISTRY.keys())

    print("=" * 60)
    print("  YOGA POSE DETECTION — TRAINING PIPELINE")
    print("=" * 60)
    config.print_config()

    print(f"\n  Models to train: {', '.join(selected_models)}")
    print(f"  GPU available : {'Yes' if tf.config.list_physical_devices('GPU') else 'No'}")

    # --- Load dataset ---
    print("\n[1/6] Loading dataset...")
    paths, labels, class_names = load_dataset_paths()
    print(f"  Total images: {len(paths)} across {len(class_names)} classes")
    for i, name in enumerate(class_names):
        count = int((labels == i).sum())
        print(f"    {name:<20s} {count:>4d} images")

    # --- Stratified split ---
    print("\n[2/6] Stratified train/val/test split...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = stratified_split(
        paths, labels
    )
    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # --- Class weights ---
    class_weights = compute_class_weights(train_labels)
    print(f"\n[3/6] Class weights: {class_weights}")

    # --- Load images into memory ---
    print("\n[4/6] Loading and preprocessing images...")
    train_images = load_images(train_paths)
    val_images = load_images(val_paths)
    test_images = load_images(test_paths)
    print(f"  Train images: {train_images.shape}")
    print(f"  Val images  : {val_images.shape}")
    print(f"  Test images : {test_images.shape}")

    num_classes = len(class_names)

    # --- Save labels ---
    save_labels(class_names)

    best_val_acc = 0
    best_model_name = ""
    results_primary = None

    # --- Train each selected model ---
    for model_name in selected_models:
        if model_name not in MODEL_REGISTRY:
            print(f"\n  [SKIP] Unknown model: {model_name}")
            continue

        info = MODEL_REGISTRY[model_name]
        preprocess_fn = info["preprocess_fn"]

        # Build data pipelines
        train_ds = build_tf_dataset(
            train_images, train_labels, preprocess_fn, num_classes, augment=True,
        )
        val_ds = build_tf_dataset(
            val_images, val_labels, preprocess_fn, num_classes, augment=False,
        )

        # Train
        model, history, hp1, hp2 = train_model(
            model_name, train_ds, val_ds, class_weights,
            num_classes, info["save_path"],
        )

        # Save plots
        save_plots(history, model_name.upper())
        save_training_history(history, model_name)

        # Evaluate on test set
        print(f"\n[5/6] Evaluating {model_name} on held-out test set...")
        results, y_pred, y_true = evaluate_on_test(
            model, test_images, test_labels, preprocess_fn, class_names,
        )

        if model_name == "efficientnetb0":
            results_primary = results
            save_confusion_matrix(
                np.array(results["confusion_matrix"]), class_names,
                f"EfficientNetB0 — Confusion Matrix",
            )
            save_classification_report(results["classification_report"], class_names)

        # Track best
        val_acc = max(history["val_accuracy"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_name = model_name

        print(f"\n  Best validation accuracy for {model_name}: {val_acc:.4f}")

    # --- Train backup models ---
    results_backup = None
    if "mobilenetv2" in selected_models:
        # Build separate test pipeline for MobileNetV2
        preprocess_fn_mv2 = MODEL_REGISTRY["mobilenetv2"]["preprocess_fn"]
        test_ds_mv2 = build_tf_dataset(
            test_images, test_labels, preprocess_fn_mv2, num_classes, augment=False,
        )
        mv2_model = tf.keras.models.load_model(config.MODEL_BACKUP_PATH)
        results_backup, _, _ = evaluate_on_test(
            mv2_model, test_images, test_labels, preprocess_fn_mv2, class_names,
        )

    save_evaluation_metrics(results_primary, results_backup)

    # --- Summary ---
    print(f"\n[6/6] Training complete!")
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Best model          : {best_model_name}")
    print(f"  Best val accuracy   : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    if results_primary:
        print(f"  Test accuracy       : {results_primary['accuracy']:.4f} ({results_primary['accuracy']*100:.2f}%)")
        print(f"  Test F1 score       : {results_primary['f1_score']:.4f}")
    print(f"\n  Outputs:")
    print(f"    Model primary     : {config.MODEL_PATH}")
    print(f"    Model backup      : {config.MODEL_BACKUP_PATH}")
    print(f"    Labels            : {config.LABELS_PATH}")
    print(f"    Confusion matrix  : {config.CONFUSION_MATRIX_PATH}")
    print(f"    Training plot     : {config.TRAINING_HISTORY_PATH}")
    print(f"    Class report      : {config.CLASSIFICATION_REPORT_PATH}")
    print(f"    Metrics JSON      : {config.EVALUATION_METRICS_PATH}")
    print(f"{'='*60}")

    return results_primary, results_backup


def main():
    parser = argparse.ArgumentParser(description="Train yoga pose classification models.")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "efficientnetb0", "mobilenetv2"],
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--skip-finetune", action="store_true",
        help="Skip phase 2 fine-tuning",
    )
    args = parser.parse_args()

    if args.model == "all":
        models = list(MODEL_REGISTRY.keys())
    else:
        models = [args.model]

    train(models, skip_finetune=args.skip_finetune)


if __name__ == "__main__":
    main()
