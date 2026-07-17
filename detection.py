"""
detection.py
============
MediaPipe-based body keypoint (skeleton) detection.

Uses the new MediaPipe Tasks API (mp.tasks.vision.PoseLandmarker)
which replaced mp.solutions in MediaPipe 0.10.x.

All image inputs/outputs use the BGR format (OpenCV default).
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

import config

# Model quality tiers: lite (fastest) < full < heavy (most accurate)
POSE_MODEL_TIERS = {
    "lite": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/latest/"
            "pose_landmarker_lite.task"
        ),
        "filename": "pose_landmarker_lite.task",
    },
    "full": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/latest/"
            "pose_landmarker_full.task"
        ),
        "filename": "pose_landmarker_full.task",
    },
    "heavy": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_heavy/float16/latest/"
            "pose_landmarker_heavy.task"
        ),
        "filename": "pose_landmarker_heavy.task",
    },
}


def _ensure_model(quality: str = "full") -> str:
    """Download the pose landmarker model if not present. Return local path.

    Parameters
    ----------
    quality : str
        One of 'lite', 'full', 'heavy'. Default 'full' for better accuracy.
    """
    quality = quality.lower()
    if quality not in POSE_MODEL_TIERS:
        quality = "full"
    info = POSE_MODEL_TIERS[quality]
    model_dir = os.path.join(config.BASE_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, info["filename"])
    if not os.path.isfile(model_path):
        print(f"Downloading MediaPipe pose model ({quality}) to {model_path} ...")
        urllib.request.urlretrieve(info["url"], model_path)
        print("Download complete.")
    return model_path


class PoseDetector:
    """
    Wrapper around MediaPipe's PoseLandmarker (Tasks API).

    Usage:
        detector = PoseDetector()
        frame, found = detector.find_pose(frame, draw=True)
        detector.close()
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        detection_confidence: float = config.MP_DETECTION_CONFIDENCE,
        tracking_confidence: float = config.MP_TRACKING_CONFIDENCE,
        model_quality: str = "full",
    ):
        model_path = _ensure_model(quality=model_quality)
        running_mode = (
            mp.tasks.vision.RunningMode.IMAGE
            if static_image_mode
            else mp.tasks.vision.RunningMode.VIDEO
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            min_pose_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.results = None
        self._running_mode = running_mode
        self._frame_count = 0

    def _detect(self, frame: np.ndarray):
        """Run detection on a BGR frame; store and return the result."""
        if frame is None or frame.size == 0:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._running_mode == mp.tasks.vision.RunningMode.VIDEO:
            self._frame_count += 1
            timestamp_ms = int(self._frame_count * 33.33)
            self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            self.results = self.detector.detect(mp_image)
        return self.results

    def find_pose(self, frame: np.ndarray, draw: bool = True):
        if frame is None or frame.size == 0:
            return frame, False

        result = self._detect(frame)
        found = bool(result and result.pose_landmarks)

        if found and draw:
            landmarks = result.pose_landmarks[0]
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame,
                landmarks,
                connections=mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=mp.tasks.vision.drawing_styles.get_default_pose_landmarks_style(),
            )

        return frame, found

    def get_landmarks(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return None

        result = self._detect(frame)
        if not result or not result.pose_landmarks:
            return None

        return [
            (lm.x, lm.y, lm.z, lm.visibility)
            for lm in result.pose_landmarks[0]
        ]

    def draw_on_image(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        annotated = image.copy()
        annotated, _ = self.find_pose(annotated, draw=True)
        return annotated

    def close(self):
        try:
            self.detector.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detection.py <image_path>")
        sys.exit(0)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: could not read image '{img_path}'")
        sys.exit(1)

    with PoseDetector(static_image_mode=True) as det:
        out, ok = det.find_pose(img, draw=True)
        print("Pose found:", ok)
        out_path = "detection_output.jpg"
        cv2.imwrite(out_path, out)
        print(f"Saved annotated image to {out_path}")
