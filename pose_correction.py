"""
pose_correction.py
==================
Pose correction module using MediaPipe landmarks and joint angle analysis.

Computes body joint angles from detected landmarks and compares them against
ideal pose templates to generate actionable correction feedback.

MediaPipe Pose Landmarks (33 landmarks):
    0: nose          11: left_shoulder     23: left_hip
    1: left_eye_inner 12: right_shoulder   24: right_hip
    2: left_eye       13: left_elbow       25: left_knee
    3: left_eye_outer 14: right_elbow      26: right_knee
    4: right_eye_inner 15: left_wrist      27: left_ankle
    5: right_eye       16: right_wrist     28: right_ankle
    6: right_eye_outer 17: left_pinky      29: left_heel
    7: left_ear        18: right_pinky     30: right_heel
    8: right_ear       19: left_index      31: left_foot_index
    9: mouth_left      20: right_index     32: right_foot_index
    10: mouth_right     21: left_thumb
                        22: right_thumb

Key joint pairs for angle calculation:
    - Elbow angle: shoulder -> elbow -> wrist
    - Knee angle: hip -> knee -> ankle
    - Hip angle: shoulder -> hip -> knee
    - Shoulder angle: elbow -> shoulder -> hip
    - Torso angle: shoulder -> hip -> vertical
    - Wrist angle: elbow -> wrist -> finger
"""

import math
import numpy as np

import config

# MediaPipe landmark indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT, RIGHT_FOOT = 31, 32

# Key joint connections for drawing
SKELETON_CONNECTIONS = [
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (NOSE, LEFT_SHOULDER), (NOSE, RIGHT_SHOULDER),
]

# ===================================================================
# ANGLE CALCULATION
# ===================================================================

def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    Calculate the angle (in degrees) at joint b between vectors ba and bc.
    a, b, c are (x, y) or (x, y, z) tuples.
    """
    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)

    ba = a - b
    bc = c - b

    dot = np.dot(ba, bc)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)

    if norm == 0:
        return 0.0

    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def extract_landmark_point(landmarks: list, idx: int) -> tuple:
    """Extract (x, y) point from landmarks list at given index."""
    if landmarks is None or idx >= len(landmarks):
        return (0.0, 0.0)
    return (landmarks[idx][0], landmarks[idx][1])


# ===================================================================
# JOINT ANGLE EXTRACTION
# ===================================================================

def extract_all_joint_angles(landmarks: list) -> dict:
    """
    Extract all relevant joint angles from a set of MediaPipe landmarks.

    Returns a dict of angle_name -> degrees.
    Returns empty dict if landmarks is None or has no pose.
    """
    if landmarks is None or len(landmarks) < 33:
        return {}

    angles = {}

    # Left arm
    angles["left_shoulder"] = calculate_angle(
        extract_landmark_point(landmarks, LEFT_ELBOW),
        extract_landmark_point(landmarks, LEFT_SHOULDER),
        extract_landmark_point(landmarks, LEFT_HIP),
    )
    angles["left_elbow"] = calculate_angle(
        extract_landmark_point(landmarks, LEFT_SHOULDER),
        extract_landmark_point(landmarks, LEFT_ELBOW),
        extract_landmark_point(landmarks, LEFT_WRIST),
    )

    # Right arm
    angles["right_shoulder"] = calculate_angle(
        extract_landmark_point(landmarks, RIGHT_ELBOW),
        extract_landmark_point(landmarks, RIGHT_SHOULDER),
        extract_landmark_point(landmarks, RIGHT_HIP),
    )
    angles["right_elbow"] = calculate_angle(
        extract_landmark_point(landmarks, RIGHT_SHOULDER),
        extract_landmark_point(landmarks, RIGHT_ELBOW),
        extract_landmark_point(landmarks, RIGHT_WRIST),
    )

    # Left leg
    angles["left_hip"] = calculate_angle(
        extract_landmark_point(landmarks, LEFT_SHOULDER),
        extract_landmark_point(landmarks, LEFT_HIP),
        extract_landmark_point(landmarks, LEFT_KNEE),
    )
    angles["left_knee"] = calculate_angle(
        extract_landmark_point(landmarks, LEFT_HIP),
        extract_landmark_point(landmarks, LEFT_KNEE),
        extract_landmark_point(landmarks, LEFT_ANKLE),
    )

    # Right leg
    angles["right_hip"] = calculate_angle(
        extract_landmark_point(landmarks, RIGHT_SHOULDER),
        extract_landmark_point(landmarks, RIGHT_HIP),
        extract_landmark_point(landmarks, RIGHT_KNEE),
    )
    angles["right_knee"] = calculate_angle(
        extract_landmark_point(landmarks, RIGHT_HIP),
        extract_landmark_point(landmarks, RIGHT_KNEE),
        extract_landmark_point(landmarks, RIGHT_ANKLE),
    )

    # Torso angle (relative to vertical)
    left_shoulder_pt = extract_landmark_point(landmarks, LEFT_SHOULDER)
    left_hip_pt = extract_landmark_point(landmarks, LEFT_HIP)
    vertical_ref = (left_hip_pt[0], left_hip_pt[1] - 100)  # point directly above hip
    angles["torso"] = calculate_angle(
        left_shoulder_pt, left_hip_pt, vertical_ref
    )

    return angles


# ===================================================================
# IDEAL POSE TEMPLATES (angles in degrees)
# ===================================================================

# For each pose, define expected joint angles with tolerances.
# Values are approximate guidelines for common yoga poses.
IDEAL_POSE_ANGLES = {
    "Boat Pose": {
        "left_knee": 170, "right_knee": 170,
        "left_hip": 135, "right_hip": 135,
        "left_shoulder": 170, "right_shoulder": 170,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 60,
    },
    "Bridge Pose": {
        "left_knee": 90, "right_knee": 90,
        "left_hip": 170, "right_hip": 170,
        "left_shoulder": 30, "right_shoulder": 30,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 170,
    },
    "Camel Pose": {
        "left_knee": 175, "right_knee": 175,
        "left_hip": 170, "right_hip": 170,
        "left_shoulder": 30, "right_shoulder": 30,
        "left_elbow": 160, "right_elbow": 160,
        "torso": 160,
    },
    "Cat Pose": {
        "left_knee": 90, "right_knee": 90,
        "left_hip": 140, "right_hip": 140,
        "left_shoulder": 130, "right_shoulder": 130,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 130,
    },
    "Chair Pose": {
        "left_knee": 90, "right_knee": 90,
        "left_hip": 110, "right_hip": 110,
        "left_shoulder": 160, "right_shoulder": 160,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 80,
    },
    "Child Pose": {
        "left_knee": 30, "right_knee": 30,
        "left_hip": 30, "right_hip": 30,
        "left_shoulder": 30, "right_shoulder": 30,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 20,
    },
    "Cobra Pose": {
        "left_knee": 175, "right_knee": 175,
        "left_hip": 175, "right_hip": 175,
        "left_shoulder": 120, "right_shoulder": 120,
        "left_elbow": 160, "right_elbow": 160,
        "torso": 140,
    },
    "Corpse Pose": {
        "left_knee": 175, "right_knee": 175,
        "left_hip": 175, "right_hip": 175,
        "left_shoulder": 175, "right_shoulder": 175,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 175,
    },
    "Cow Pose": {
        "left_knee": 110, "right_knee": 110,
        "left_hip": 160, "right_hip": 160,
        "left_shoulder": 110, "right_shoulder": 110,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 110,
    },
    "Cross Legged Pose": {
        "left_knee": 45, "right_knee": 45,
        "left_hip": 135, "right_hip": 135,
        "left_shoulder": 170, "right_shoulder": 170,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 175,
    },
    "Downward Dog": {
        "left_knee": 170, "right_knee": 170,
        "left_hip": 60, "right_hip": 60,
        "left_shoulder": 170, "right_shoulder": 170,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 45,
    },
    "Goddess Pose": {
        "left_knee": 90, "right_knee": 90,
        "left_hip": 135, "right_hip": 135,
        "left_shoulder": 90, "right_shoulder": 90,
        "left_elbow": 90, "right_elbow": 90,
        "torso": 175,
    },
    "Mountain Pose": {
        "left_knee": 175, "right_knee": 175,
        "left_hip": 175, "right_hip": 175,
        "left_shoulder": 170, "right_shoulder": 170,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 175,
    },
    "Plank Pose": {
        "left_knee": 175, "right_knee": 175,
        "left_hip": 175, "right_hip": 175,
        "left_shoulder": 90, "right_shoulder": 90,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 175,
    },
    "Tree Pose": {
        "left_knee": 175, "right_knee": 45,
        "left_hip": 175, "right_hip": 90,
        "left_shoulder": 170, "right_shoulder": 170,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 175,
    },
    "Triangle Pose": {
        "left_knee": 175, "right_knee": 175,
        "left_hip": 150, "right_hip": 150,
        "left_shoulder": 170, "right_shoulder": 170,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 160,
    },
    "Warrior 2": {
        "left_knee": 90, "right_knee": 170,
        "left_hip": 135, "right_hip": 170,
        "left_shoulder": 170, "right_shoulder": 170,
        "left_elbow": 175, "right_elbow": 175,
        "torso": 175,
    },
}

# Human-readable names for joint angles
JOINT_LABELS = {
    "left_knee": "Left knee", "right_knee": "Right knee",
    "left_hip": "Left hip", "right_hip": "Right hip",
    "left_shoulder": "Left shoulder", "right_shoulder": "Right shoulder",
    "left_elbow": "Left elbow", "right_elbow": "Right elbow",
    "torso": "Torso",
}


# ===================================================================
# CORRECTION FEEDBACK GENERATION
# ===================================================================

def analyze_pose_angles(landmarks: list, pose_name: str) -> list[dict]:
    """
    Compare detected joint angles against ideal pose template.

    Parameters
    ----------
    landmarks : list of (x, y, z, visibility) tuples from MediaPipe
    pose_name : str — predicted pose name

    Returns
    -------
    list of dicts, each with:
        joint, detected_angle, ideal_angle, deviation, feedback, severity
    """
    detected = extract_all_joint_angles(landmarks)
    if not detected:
        return []

    template = IDEAL_POSE_ANGLES.get(pose_name, {})
    if not template:
        return []

    tolerance = config.ANGLE_TOLERANCE
    corrections = []

    # Map detected angle names to ideal angle names
    for angle_name, ideal_val in template.items():
        detected_val = detected.get(angle_name)
        if detected_val is None:
            continue

        deviation = detected_val - ideal_val
        abs_dev = abs(deviation)

        if abs_dev <= tolerance:
            severity = "good"
            feedback = f"{JOINT_LABELS.get(angle_name, angle_name)} is good ({detected_val:.0f}°)."
        elif abs_dev <= tolerance * 2:
            severity = "minor"
            direction = "straighten" if deviation > 0 else "bend more"
            feedback = (
                f"{JOINT_LABELS.get(angle_name, angle_name)}: "
                f"{direction} by {abs_dev:.0f}° (currently {detected_val:.0f}°, ideal {ideal_val:.0f}°)."
            )
        else:
            severity = "major"
            direction = "straighten" if deviation > 0 else "bend more"
            feedback = (
                f"{JOINT_LABELS.get(angle_name, angle_name)}: "
                f"significantly {direction} by {abs_dev:.0f}° "
                f"(currently {detected_val:.0f}°, ideal {ideal_val:.0f}°)."
            )

        corrections.append({
            "joint": angle_name,
            "detected_angle": round(detected_val, 1),
            "ideal_angle": ideal_val,
            "deviation": round(deviation, 1),
            "feedback": feedback,
            "severity": severity,
        })

    # Sort by severity (major first)
    severity_order = {"major": 0, "minor": 1, "good": 2}
    corrections.sort(key=lambda x: severity_order.get(x["severity"], 3))

    return corrections


def generate_correction_summary(corrections: list[dict]) -> str:
    """
    Generate a concise human-readable summary from joint angle corrections.
    """
    if not corrections:
        return "No pose detected for correction analysis."

    major = [c for c in corrections if c["severity"] == "major"]
    minor = [c for c in corrections if c["severity"] == "minor"]
    good = [c for c in corrections if c["severity"] == "good"]

    lines = []

    if major:
        lines.append("Needs significant correction:")
        for c in major[:3]:
            lines.append(f"  - {c['feedback']}")

    if minor:
        lines.append("Minor adjustments:")
        for c in minor[:3]:
            lines.append(f"  - {c['feedback']}")

    if good and not major:
        lines.append(f"Alignment looks good! ({len(good)} joints within range)")

    if not major and not minor and good:
        lines.append("Excellent pose! All joints are well aligned.")

    return "\n".join(lines)


def build_complete_correction(
    landmarks: list, pose_name: str, confidence: float
) -> dict:
    """
    Build complete correction feedback including angle analysis and pose tips.

    Returns
    -------
    dict with keys: corrections, summary, tip, confidence, pose
    """
    corrections = analyze_pose_angles(landmarks, pose_name)
    summary = generate_correction_summary(corrections)

    from feedback import get_pose_tip
    tip = get_pose_tip(pose_name)

    return {
        "pose": pose_name,
        "confidence": confidence,
        "corrections": corrections,
        "summary": summary,
        "tip": tip,
    }


# ===================================================================
# SELF-TEST
# ===================================================================
if __name__ == "__main__":
    print("Pose Correction Module")
    print("=" * 60)
    for pose in config.POSES:
        angles = IDEAL_POSE_ANGLES.get(pose, {})
        print(f"\n{pose}:")
        for joint, val in angles.items():
            print(f"  {JOINT_LABELS.get(joint, joint):20s} {val:3d}°")

    print("\n\nAngle calculation test:")
    a, b, c = (0, 0), (1, 0), (1, 1)
    angle = calculate_angle(a, b, c)
    print(f"  Angle at (1,0) between (0,0)-(1,0)-(1,1): {angle:.1f}° (expected 90.0°)")
