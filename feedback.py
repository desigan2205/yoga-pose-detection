"""
feedback.py
===========
Confidence-based feedback and instructional tips for yoga poses.

Provides:
    - get_feedback(confidence)  -> message, level, color, emoji
    - get_pose_tip(pose_name)   -> short alignment tip
    - build_full_feedback()     -> combined dict
"""

import config

POSE_TIPS = {
    "Boat Pose": (
        "Balance on your sit bones, lift legs to 45°, extend arms parallel "
        "to the floor, and keep your chest lifted."
    ),
    "Bridge Pose": (
        "Lie on your back, bend knees feet on floor, lift hips high, "
        "interlace hands under your back, press shoulders down."
    ),
    "Camel Pose": (
        "Kneel with knees hip-width apart, lean back reaching for heels, "
        "open your chest, keep hips over knees."
    ),
    "Cat Pose": (
        "On all fours, exhale as you round your spine like a cat, "
        "tuck your chin, and draw your belly in."
    ),
    "Chair Pose": (
        "Bend knees as if sitting in a chair, keep weight in heels, "
        "reach arms overhead with palms facing each other."
    ),
    "Child Pose": (
        "Kneel and sit back on heels, extend arms forward on the mat, "
        "rest your forehead down, and breathe deeply."
    ),
    "Cobra Pose": (
        "Lie face down, place hands under shoulders, gently lift chest "
        "while keeping elbows close and shoulders relaxed."
    ),
    "Corpse Pose": (
        "Lie flat on your back, arms at sides palms up, legs slightly apart, "
        "close your eyes and relax every muscle."
    ),
    "Cow Pose": (
        "On all fours, inhale as you arch your spine, lift your chest "
        "and tailbone, and let your belly drop."
    ),
    "Cross Legged Pose": (
        "Sit with legs crossed, spine tall, hands on knees, "
        "relax your shoulders and breathe steadily."
    ),
    "Downward Dog": (
        "From all fours, lift hips up and back, straighten legs, "
        "press heels toward the mat, and lengthen the spine."
    ),
    "Goddess Pose": (
        "Stand with feet wide, turn toes out, bend knees to 90°, "
        "keep pelvis tucked and arms raised with bent elbows."
    ),
    "Mountain Pose": (
        "Stand tall with feet together, arms at sides, "
        "engage thighs, lift chest, and ground through your feet."
    ),
    "Plank Pose": (
        "Keep your body in a straight line from head to heels, "
        "engage your core, and keep shoulders over wrists."
    ),
    "Tree Pose": (
        "Balance on one foot, place the sole of the other foot on "
        "your inner thigh or calf (never the knee), hands at heart center."
    ),
    "Triangle Pose": (
        "Stand with legs apart, extend arms, hinge at the hip to reach "
        "one hand to your shin, the other arm straight up."
    ),
    "Warrior 2": (
        "Front knee bent at 90°, back leg straight, arms parallel "
        "to the floor, gaze over the front hand."
    ),
}

DEFAULT_TIP = "Hold the pose, breathe deeply, and keep your form steady."


def get_feedback(confidence: float) -> dict:
    """
    Return feedback dict based on confidence score.

    Parameters
    ----------
    confidence : float (0.0 - 1.0)

    Returns
    -------
    dict with keys: message, level, color, emoji
    """
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if confidence >= config.CONF_HIGH:
        return {
            "message": "Great pose! Hold it.",
            "level": "high",
            "color": "#16a34a",
            "emoji": "✅",
        }
    elif confidence >= config.CONF_MEDIUM:
        return {
            "message": "Almost there! Adjust slightly.",
            "level": "medium",
            "color": "#d97706",
            "emoji": "⚠️",
        }
    else:
        return {
            "message": "Try again. Check your alignment.",
            "level": "low",
            "color": "#dc2626",
            "emoji": "❌",
        }


def get_pose_tip(pose_name: str) -> str:
    """Return a short alignment tip for the given pose."""
    if not pose_name:
        return DEFAULT_TIP
    return POSE_TIPS.get(pose_name, DEFAULT_TIP)


def build_full_feedback(pose_name: str, confidence: float) -> dict:
    """
    Combine confidence feedback + pose tip into one dict.

    Returns dict with keys: message, level, color, emoji, tip, pose, confidence
    """
    fb = get_feedback(confidence)
    fb["tip"] = get_pose_tip(pose_name)
    fb["pose"] = pose_name
    fb["confidence"] = confidence
    return fb


if __name__ == "__main__":
    for c in [0.95, 0.70, 0.40]:
        fb = build_full_feedback("Tree Pose", c)
        print(f"conf={c:.2f} -> {fb['emoji']} {fb['message']} | tip: {fb['tip'][:50]}...")
