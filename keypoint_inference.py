"""
keypoint_inference.py
=====================
Inference module for the Keypoint-Only model (fusion_keypoint_best.keras).

Uses MediaPipe (detection.py) to extract 33 landmarks, converts to a 204-dim
feature vector (raw 99 + engineered 105: angles, distances, center-relative,
symmetry), and runs the keypoint classifier (HP-tuned, 89.76% accuracy).

Public API:
    KeypointPredictor            — loads model, predicts from image or landmarks
    KeypointPredictor.predict    — (pose_name, confidence, top3)
    get_keypoint_predictor()     — cached singleton for Streamlit
"""

import os
import json

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import config
from detection import PoseDetector
from keypoint_features import extract_combined_features

FEATURE_SIZE = 204
SPECIALIST_CLASSES = ["Cat Pose", "Cross Legged Pose"]

FUSION_CLASSES = [
    "Boat Pose", "Bridge Pose", "Camel Pose", "Cat Pose",
    "Chair Pose", "Child Pose", "Cobra Pose", "Corpse Pose",
    "Cow Pose", "Cross Legged Pose", "Downward Dog",
    "Goddess Pose", "Mountain Pose", "Plank Pose",
    "Tree Pose", "Triangle Pose", "Warrior 2",
]


class KeypointPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(
            config.MODEL_DIR, "fusion_keypoint_best.keras"
        )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Keypoint model not found: {self.model_path}")
        self.model = load_model(self.model_path, compile=False)
        self.labels = {i: name for i, name in enumerate(FUSION_CLASSES)}

        specialist_path = os.path.join(config.MODEL_DIR, "cat_cross_model.keras")
        self.specialist = None
        if os.path.exists(specialist_path):
            self.specialist = load_model(specialist_path, compile=False)
            self.specialist_labels = {0: "Cat Pose", 1: "Cross Legged Pose"}

    def landmarks_to_features(self, landmarks):
        return extract_combined_features(landmarks)

    def _specialist_features(self, landmarks):
        lm = np.asarray(landmarks)
        arr = lm.flatten()
        cx = (lm[23, 0] + lm[24, 0]) / 2
        cy = (lm[23, 1] + lm[24, 1]) / 2
        arr2 = np.array([[lm[i, 0] - cx, lm[i, 1] - cy, lm[i, 2], lm[i, 3]]
                        for i in range(len(lm))]).flatten()
        return np.concatenate([arr, arr2])

    def predict(self, image=None, landmarks=None, top_k=3):
        if landmarks is None:
            if image is None:
                return "No input", 0.0, []
            with PoseDetector(static_image_mode=True) as det:
                landmarks = det.get_landmarks(image)
            if landmarks is None:
                return "No person detected", 0.0, []

        feats = self.landmarks_to_features(landmarks)
        batch = np.expand_dims(feats, axis=0)
        preds = self.model.predict(batch, verbose=0)[0]

        best_idx = int(np.argmax(preds))
        best_name = self.labels.get(best_idx, f"Class {best_idx}")
        best_conf = float(preds[best_idx])
        top_indices = preds.argsort()[::-1][:top_k]

        if self.specialist is not None and best_name in SPECIALIST_CLASSES:
            s_feats = self._specialist_features(landmarks)
            s_batch = np.expand_dims(s_feats, axis=0)
            s_preds = self.specialist.predict(s_batch, verbose=0)[0]
            s_idx = int(np.argmax(s_preds))
            s_name = self.specialist_labels.get(s_idx, best_name)
            s_conf = float(s_preds[s_idx])
            if s_name in SPECIALIST_CLASSES and s_conf > best_conf:
                best_name = s_name
                best_conf = s_conf

        top_list = [
            (self.labels.get(int(i), f"Class {i}"), float(preds[int(i)]))
            for i in top_indices
        ]

        return best_name, best_conf, top_list


_predictor_instance = None


def get_keypoint_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = KeypointPredictor()
    return _predictor_instance


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python keypoint_inference.py <image_path>")
        sys.exit(0)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"ERROR: could not read '{sys.argv[1]}'")
        sys.exit(1)

    try:
        predictor = get_keypoint_predictor()
        name, conf, top3 = predictor.predict(img)
        print(f"\nPredicted pose : {name}")
        print(f"Confidence     : {conf * 100:.2f}%")
        print("\nTop 3:")
        for n, p in top3:
            print(f"   {n:20s} {p * 100:6.2f}%")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
