"""
model_predict.py
================
Inference module for yoga pose classification.

Supports both EfficientNetB0 (primary) and MobileNetV2 (backup) models.

Public API:
    YogaPredictor           — class that loads model + labels once
    YogaPredictor.predict   — returns (pose_name, confidence, top3)
    get_predictor()         — cached singleton for Streamlit
"""

import os
import json

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import config


# Map model names to their preprocessing functions
PREPROCESS_FN_MAP = {
    "efficientnetv2": tf.keras.applications.efficientnet_v2.preprocess_input,
    "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
}


def _detect_model_type(model_path: str) -> str:
    """Try to determine which architecture the saved model uses."""
    try:
        loaded = load_model(model_path, compile=False)
        input_name = loaded.inputs[0].name.lower()
        if "efficientnetv2" in input_name:
            return "efficientnetv2"
        elif "efficientnet" in input_name:
            return "efficientnetb0"
        elif "mobilenetv2" in input_name or "mobilenet" in input_name:
            return "mobilenetv2"
        elif "resnet" in input_name:
            return "resnet50"
        # Fallback: check path
        path_lower = model_path.lower()
        if "efficientnetv2" in path_lower:
            return "efficientnetv2"
        if "mobilenetv2" in path_lower:
            return "mobilenetv2"
        if "resnet50" in path_lower:
            return "resnet50"
        if "efficientnetb0" in path_lower:
            return "efficientnetb0"
        # Fallback: check layer names of the loaded model
        for layer in loaded.layers:
            lname = layer.name.lower()
            if "mobilenetv2" in lname or "mobilenet" in lname:
                return "mobilenetv2"
            if "efficientnet" in lname:
                return "efficientnetb0"
            if "resnet" in lname:
                return "resnet50"
        return "mobilenetv2"
    except Exception:
        return "efficientnetv2"


class YogaPredictor:
    """
    Loads a trained Keras model and class labels, then predicts yoga poses.

    Usage:
        predictor = YogaPredictor()
        name, conf, top3 = predictor.predict(bgr_image)

    Automatically falls back to MobileNetV2 if primary model is missing.
    """

    def __init__(self, model_path: str = None, labels_path: str = None):
        self.model_path = model_path
        self.labels_path = labels_path or config.LABELS_PATH

        self.model = None
        self.labels = None
        self.preprocess_fn = None

        self._load()

    def _load(self):
        """Load model and labels with fallback support."""
        # Priority: BEST_MODEL_PATH > MODEL_PATH > MODEL_BACKUP_PATH
        candidates = []
        if self.model_path:
            candidates.append(self.model_path)
        candidates.append(config.BEST_MODEL_PATH)
        candidates.append(config.MODEL_PATH)
        candidates.append(config.MODEL_BACKUP_PATH)

        for candidate in candidates:
            if os.path.exists(candidate):
                try:
                    model_type = _detect_model_type(candidate)
                    self.preprocess_fn = PREPROCESS_FN_MAP.get(
                        model_type,
                        tf.keras.applications.efficientnet_v2.preprocess_input,
                    )
                    self.model = load_model(candidate, compile=False)
                    self.model_path = candidate
                    break
                except Exception as exc:
                    print(f"Warning: failed to load {candidate}: {exc}")
                    continue

        if self.model is None:
            raise RuntimeError(
                f"No trained model found. Tried:\n"
                + "\n".join(f"  - {p}" for p in candidates)
                + "\n\nPlease train the model first:\n"
                "    python train_best.py"
            )

        # --- Load labels ---
        if not os.path.exists(self.labels_path):
            raise RuntimeError(
                f"Labels file not found: {self.labels_path}\n"
                "Re-run: python model_train.py"
            )
        try:
            with open(self.labels_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.labels = {int(k): v for k, v in raw.items()}
        except Exception as exc:
            raise RuntimeError(f"Failed to load labels: {exc}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize + preprocess a BGR image into a model-ready batch."""
        if image is None or image.size == 0:
            raise ValueError("Empty image passed to predictor.")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (config.IMG_WIDTH, config.IMG_HEIGHT))
        arr = np.expand_dims(resized.astype(np.float32), axis=0)
        arr = self.preprocess_fn(arr)
        return arr

    def predict(self, image: np.ndarray, top_k: int = 3):
        """
        Predict the yoga pose for a single BGR image.

        Returns
        -------
        (pose_name, confidence, top_k_list)
            pose_name   : str   — best predicted pose
            confidence  : float — probability of the best pose (0-1)
            top_k_list  : list[(name, prob)] sorted high->low
        """
        batch = self._preprocess(image)
        preds = self.model.predict(batch, verbose=0)[0]

        best_idx = int(np.argmax(preds))
        best_name = self.labels.get(best_idx, f"Class {best_idx}")
        best_conf = float(preds[best_idx])

        top_indices = preds.argsort()[::-1][:top_k]
        top_list = [
            (self.labels.get(int(i), f"Class {i}"), float(preds[int(i)]))
            for i in top_indices
        ]

        return best_name, best_conf, top_list


# -------------------------------------------------------------------
# Module-level singleton helper (caches the model across Streamlit reruns)
# -------------------------------------------------------------------
_predictor_instance = None


def get_predictor() -> YogaPredictor:
    """Return a cached YogaPredictor instance (loads model once)."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = YogaPredictor()
    return _predictor_instance


# -------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_predict.py <image_path>")
        sys.exit(0)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"ERROR: could not read image '{sys.argv[1]}'")
        sys.exit(1)

    try:
        predictor = get_predictor()
        name, conf, top3 = predictor.predict(img)
        print(f"\nPredicted pose : {name}")
        print(f"Confidence     : {conf * 100:.2f}%")
        print("\nTop 3:")
        for n, p in top3:
            print(f"   {n:20s} {p * 100:6.2f}%")
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
