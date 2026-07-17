"""
evaluate.py
===========
Standalone evaluation module for trained yoga pose classification models.

Loads a trained model and runs comprehensive evaluation on the test set:

    - Accuracy, Precision, Recall, F1 Score
    - Confusion Matrix (saved as PNG)
    - Classification Report (saved as TXT)
    - Per-class metrics breakdown

Usage:
    python evaluate.py                          # evaluate primary model (EfficientNetB0)
    python evaluate.py --model mobilenetv2      # evaluate MobileNetV2
    python evaluate.py --model all              # evaluate all available models
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

import config
from data_processing import load_dataset_paths, stratified_split, load_images

tf.get_logger().setLevel("ERROR")


def load_model_and_preprocess(model_path: str):
    """Load a Keras model and determine its preprocessing function."""
    model = tf.keras.models.load_model(model_path, compile=False)

    input_name = model.inputs[0].name.lower()
    if "efficientnet" in input_name:
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
        model_type = "EfficientNetB0"
    elif "mobilenetv2" in input_name or "mobilenet" in input_name:
        preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
        model_type = "MobileNetV2"
    else:
        preprocess_fn = lambda x: x / 255.0
        model_type = "Unknown"

    return model, preprocess_fn, model_type


def evaluate_model(
    model, preprocess_fn, test_images, test_labels, class_names, model_name="Model"
):
    """Run comprehensive evaluation and print results."""
    test_pp = preprocess_fn(test_images)
    test_oh = tf.keras.utils.to_categorical(test_labels, len(class_names))

    loss, acc = model.evaluate(test_pp, test_oh, verbose=0)
    y_prob = model.predict(test_pp, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_labels

    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print(f"\n{'='*60}")
    print(f"  {model_name} — TEST SET EVALUATION")
    print(f"{'='*60}")
    print(f"  Test Loss      : {loss:.4f}")
    print(f"  Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision (w)  : {precision:.4f}")
    print(f"  Recall    (w)  : {recall:.4f}")
    print(f"  F1 Score  (w)  : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(cr)
    print(f"\n  Confusion Matrix:")
    print(np.array2string(cm, separator=", "))

    results = {
        "model": model_name,
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4, output_dict=True,
        ),
    }

    return results, y_pred, y_true


def save_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix -> {save_path}")


def save_classification_report_text(report_dict, class_names, save_path):
    """Save classification report as formatted text."""
    lines = [
        "=" * 70,
        "  YOGA POSE DETECTION — CLASSIFICATION REPORT",
        "=" * 70,
        "",
        f"{'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}",
        "-" * 60,
    ]
    for cls in class_names:
        if cls in report_dict:
            m = report_dict[cls]
            lines.append(
                f"{cls:<20s} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1-score']:>10.4f} {m['support']:>10.0f}"
            )
    lines.append("-" * 60)
    if "weighted avg" in report_dict:
        w = report_dict["weighted avg"]
        lines.append(
            f"{'Weighted Avg':<20s} {w['precision']:>10.4f} {w['recall']:>10.4f} "
            f"{w['f1-score']:>10.4f} {w['support']:>10.0f}"
        )
    if "accuracy" in report_dict:
        lines.append(f"{'Accuracy':<20s} {'':>10s} {'':>10s} {report_dict['accuracy']:>10.4f} {'':>10s}")
    lines.extend(["", "=" * 70])

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved classification report -> {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate yoga pose classification models.")
    parser.add_argument(
        "--model", type=str, default="primary",
        choices=["primary", "mobilenetv2", "all"],
        help="Which model to evaluate (default: primary)",
    )
    args = parser.parse_args()

    # Load test dataset
    print("Loading dataset...")
    paths, labels, class_names = load_dataset_paths()
    _, _, _, _, test_paths, test_labels = stratified_split(paths, labels)
    test_images = load_images(test_paths)
    print(f"  Test images: {len(test_paths)} across {len(class_names)} classes")

    models_to_eval = []

    if args.model in ("primary", "all"):
        if os.path.exists(config.MODEL_PATH):
            models_to_eval.append(("EfficientNetB0", config.MODEL_PATH))

    if args.model in ("mobilenetv2", "all"):
        if os.path.exists(config.MODEL_BACKUP_PATH):
            models_to_eval.append(("MobileNetV2", config.MODEL_BACKUP_PATH))

    if not models_to_eval:
        print("No trained models found. Train first: python model_train.py")
        sys.exit(1)

    all_results = []

    for display_name, model_path in models_to_eval:
        print(f"\n{'='*60}")
        print(f"  Loading {display_name}...")
        model, preprocess_fn, detected_type = load_model_and_preprocess(model_path)
        print(f"  Detected type: {detected_type}")

        results, y_pred, y_true = evaluate_model(
            model, preprocess_fn, test_images, test_labels,
            class_names, model_name=display_name,
        )
        all_results.append(results)

        # Save outputs
        model_slug = display_name.lower().replace(" ", "_")
        cm_path = os.path.join(
            config.OUTPUT_DIR, f"confusion_matrix_{model_slug}.png"
        )
        save_confusion_matrix(
            np.array(results["confusion_matrix"]), class_names,
            cm_path, title=f"{display_name} — Confusion Matrix",
        )

        cr_path = os.path.join(
            config.OUTPUT_DIR, f"classification_report_{model_slug}.txt"
        )
        save_classification_report_text(
            results["classification_report"], class_names, cr_path,
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['model']:<20s} {r['accuracy']:>8.4f}  "
            f"{r['precision']:>8.4f}  {r['recall']:>8.4f}  {r['f1_score']:>8.4f}"
        )

    # Save combined results
    with open(config.EVALUATION_METRICS_PATH, "w") as f:
        json.dump({"evaluations": all_results}, f, indent=2)
    print(f"\n  Saved all metrics -> {config.EVALUATION_METRICS_PATH}")


if __name__ == "__main__":
    main()
