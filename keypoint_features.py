import numpy as np

NUM_LANDMARKS = 33
LANDMARK_DIM = 4  # x, y, z, visibility

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT, RIGHT_FOOT = 31, 32


def _vec(a, b):
    return np.array([b[0] - a[0], b[1] - a[1]])


def _angle(p1, p2, p3):
    v1 = _vec(p2, p1)
    v2 = _vec(p2, p3)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-8:
        return 0.0
    c = np.clip(dot / norm, -1.0, 1.0)
    return float(np.arccos(c))


def _distance(a, b):
    return float(np.linalg.norm([a[0] - b[0], a[1] - b[1]]))


def extract_engineered_features(landmarks):
    landmarks = np.asarray(landmarks)
    if landmarks.shape != (NUM_LANDMARKS, LANDMARK_DIM):
        raise ValueError(f"Expected ({NUM_LANDMARKS}, {LANDMARK_DIM}), got {landmarks.shape}")

    # Only use landmarks with good visibility (threshold)
    def pt(idx):
        return landmarks[idx, 0], landmarks[idx, 1]

    features = []

    # -- Joint Angles (8) --
    # Left elbow: shoulder-elbow-wrist
    features.append(_angle(pt(LEFT_SHOULDER), pt(LEFT_ELBOW), pt(LEFT_WRIST)))
    # Right elbow
    features.append(_angle(pt(RIGHT_SHOULDER), pt(RIGHT_ELBOW), pt(RIGHT_WRIST)))
    # Left shoulder: hip-shoulder-elbow
    features.append(_angle(pt(LEFT_HIP), pt(LEFT_SHOULDER), pt(LEFT_ELBOW)))
    # Right shoulder
    features.append(_angle(pt(RIGHT_HIP), pt(RIGHT_SHOULDER), pt(RIGHT_ELBOW)))
    # Left knee: hip-knee-ankle
    features.append(_angle(pt(LEFT_HIP), pt(LEFT_KNEE), pt(LEFT_ANKLE)))
    # Right knee
    features.append(_angle(pt(RIGHT_HIP), pt(RIGHT_KNEE), pt(RIGHT_ANKLE)))
    # Left hip: shoulder-hip-knee
    features.append(_angle(pt(LEFT_SHOULDER), pt(LEFT_HIP), pt(LEFT_KNEE)))
    # Right hip
    features.append(_angle(pt(RIGHT_SHOULDER), pt(RIGHT_HIP), pt(RIGHT_KNEE)))
    # Neck angle: nose - mid_shoulder - horizontal
    mid_shoulder = ((pt(LEFT_SHOULDER)[0] + pt(RIGHT_SHOULDER)[0]) / 2,
                    (pt(LEFT_SHOULDER)[1] + pt(RIGHT_SHOULDER)[1]) / 2)
    nose = pt(NOSE)
    v_neck = np.array([nose[0] - mid_shoulder[0], nose[1] - mid_shoulder[1]])
    v_horiz = np.array([1.0, 0.0])
    norm_n = np.linalg.norm(v_neck)
    if norm_n > 1e-8:
        neck_angle = float(np.arccos(np.clip(np.dot(v_neck, v_horiz) / norm_n, -1.0, 1.0)))
    else:
        neck_angle = 0.0
    features.append(neck_angle)
    # Torso angle: mid_shoulder - mid_hip vs vertical
    mid_hip = ((pt(LEFT_HIP)[0] + pt(RIGHT_HIP)[0]) / 2,
               (pt(LEFT_HIP)[1] + pt(RIGHT_HIP)[1]) / 2)
    v_torso = np.array([mid_hip[0] - mid_shoulder[0], mid_hip[1] - mid_shoulder[1]])
    v_vert = np.array([0.0, 1.0])
    norm_t = np.linalg.norm(v_torso)
    if norm_t > 1e-8:
        torso_angle = float(np.arccos(np.clip(np.dot(v_torso, v_vert) / norm_t, -1.0, 1.0)))
    else:
        torso_angle = 0.0
    features.append(torso_angle)

    # -- Normalized Distances (normalized by torso height) --
    torso_height = _distance(mid_shoulder, mid_hip)
    if torso_height < 1e-8:
        torso_height = 1.0

    # Shoulder width
    features.append(_distance(pt(LEFT_SHOULDER), pt(RIGHT_SHOULDER)) / torso_height)
    # Hip width
    features.append(_distance(pt(LEFT_HIP), pt(RIGHT_HIP)) / torso_height)
    # Wrist distance apart
    features.append(_distance(pt(LEFT_WRIST), pt(RIGHT_WRIST)) / torso_height)
    # Ankle distance apart
    features.append(_distance(pt(LEFT_ANKLE), pt(RIGHT_ANKLE)) / torso_height)
    # Left wrist to shoulder
    features.append(_distance(pt(LEFT_WRIST), pt(LEFT_SHOULDER)) / torso_height)
    # Right wrist to shoulder
    features.append(_distance(pt(RIGHT_WRIST), pt(RIGHT_SHOULDER)) / torso_height)
    # Left ankle to hip
    features.append(_distance(pt(LEFT_ANKLE), pt(LEFT_HIP)) / torso_height)
    # Right ankle to hip
    features.append(_distance(pt(RIGHT_ANKLE), pt(RIGHT_HIP)) / torso_height)
    # Left knee to hip
    features.append(_distance(pt(LEFT_KNEE), pt(LEFT_HIP)) / torso_height)
    # Right knee to hip
    features.append(_distance(pt(RIGHT_KNEE), pt(RIGHT_HIP)) / torso_height)
    # Nose to mid_hip
    features.append(_distance(nose, mid_hip) / torso_height)
    # Wrist to ankle (left)
    features.append(_distance(pt(LEFT_WRIST), pt(LEFT_ANKLE)) / torso_height)
    # Wrist to ankle (right)
    features.append(_distance(pt(RIGHT_WRIST), pt(RIGHT_ANKLE)) / torso_height)

    # -- Center-relative coordinates (relative to mid_hip) --
    for i in range(NUM_LANDMARKS):
        features.append(landmarks[i, 0] - mid_hip[0])
        features.append(landmarks[i, 1] - mid_hip[1])

    # -- Symmetry features --
    # Left-right differences for corresponding landmarks
    pairs = [(LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_ELBOW, RIGHT_ELBOW),
             (LEFT_WRIST, RIGHT_WRIST), (LEFT_HIP, RIGHT_HIP),
             (LEFT_KNEE, RIGHT_KNEE), (LEFT_ANKLE, RIGHT_ANKLE),
             (LEFT_HEEL, RIGHT_HEEL), (LEFT_FOOT, RIGHT_FOOT)]
    for l_idx, r_idx in pairs:
        features.append(landmarks[l_idx, 0] - landmarks[r_idx, 0])
        features.append(landmarks[l_idx, 1] - landmarks[r_idx, 1])

    return np.array(features, dtype=np.float32)


def extract_combined_features(landmarks):
    raw = np.zeros(99, dtype=np.float32)
    for i, (x, y, z, vis) in enumerate(landmarks):
        raw[i * 3] = x
        raw[i * 3 + 1] = y
        raw[i * 3 + 2] = vis
    engineered = extract_engineered_features(landmarks)
    return np.concatenate([raw, engineered])


def get_feature_dim():
    dummy = np.zeros((33, 4), dtype=np.float32)
    return len(extract_combined_features(dummy))


if __name__ == "__main__":
    dummy = np.zeros((33, 4), dtype=np.float32)
    feats = extract_combined_features(dummy)
    print(f"Combined feature size: {len(feats)}")
    print(f"Raw: 99, Engineered: {len(feats) - 99}")
