import math

def distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_face_scale(landmarks):
    """
    Computes a scale reference for the face:
    Inter-ocular distance (outer eye corners: 33 and 263) or chin to forehead.
    Ensures all measurements are invariant to distance from camera.
    """
    p_eye_left = landmarks[33]
    p_eye_right = landmarks[263]
    eye_dist = distance(p_eye_left, p_eye_right)
    if eye_dist > 1e-4:
        return eye_dist

    # Fallback to forehead to chin
    p_forehead = landmarks[10]
    p_chin = landmarks[152]
    return max(distance(p_forehead, p_chin), 1e-4)

def mouth_aspect_ratio(landmarks):
    """
    Mouth Aspect Ratio (MAR):
    Ratio of vertical inner lip separation (13, 14) to horizontal mouth width (78, 308).
    Shouting / yelling produces a high MAR (> 0.38 - 0.50).
    Normal speaking is typically < 0.25.
    """
    v_dist = distance(landmarks[13], landmarks[14])
    h_dist = distance(landmarks[78], landmarks[308])
    if h_dist < 1e-4:
        return 0.0
    return v_dist / h_dist

def eyebrow_furrow_ratio(landmarks, face_scale):
    """
    Calculates inner eyebrow compression (anger / scowl):
    Distance between inner eyebrows (55, 285) normalized by face scale.
    Lower values indicate tightly furrowed, angry brows.
    Also checks brow-to-eye vertical distance (70 to 159 and 300 to 386).
    """
    inner_brow_dist = distance(landmarks[55], landmarks[285]) / face_scale
    
    # Brow to eye distance
    left_brow_eye = distance(landmarks[70], landmarks[159]) / face_scale
    right_brow_eye = distance(landmarks[300], landmarks[386]) / face_scale
    avg_brow_eye = (left_brow_eye + right_brow_eye) / 2.0

    return inner_brow_dist, avg_brow_eye

def relative_facial_motion(prev_landmarks, curr_landmarks, face_scale):
    """
    Calculates intrinsic facial deformation/motion by subtracting the overall
    head translation vector. This prevents camera panning or whole-body walking
    from triggering false facial aggression alerts.
    """
    if prev_landmarks is None or len(prev_landmarks) != len(curr_landmarks):
        return 0.0, 0.0

    # Head centroid shift
    prev_cx = sum(p[0] for p in prev_landmarks) / len(prev_landmarks)
    prev_cy = sum(p[1] for p in prev_landmarks) / len(prev_landmarks)
    curr_cx = sum(p[0] for p in curr_landmarks) / len(curr_landmarks)
    curr_cy = sum(p[1] for p in curr_landmarks) / len(curr_landmarks)

    shift_x = curr_cx - prev_cx
    shift_y = curr_cy - prev_cy
    head_shift = math.sqrt(shift_x**2 + shift_y**2) / face_scale

    # Intrinsic landmark movement (deformation relative to head shift)
    # Sample 64 key landmark points for efficiency
    sample_indices = range(0, min(len(curr_landmarks), 468), 7)
    total_rel_motion = 0.0
    for idx in sample_indices:
        p_prev = prev_landmarks[idx]
        p_curr = curr_landmarks[idx]
        rel_dx = (p_curr[0] - p_prev[0]) - shift_x
        rel_dy = (p_curr[1] - p_prev[1]) - shift_y
        total_rel_motion += math.sqrt(rel_dx**2 + rel_dy**2)

    avg_rel_motion = (total_rel_motion / len(sample_indices)) / face_scale
    return avg_rel_motion, head_shift

def estimate_head_yaw(landmarks, face_scale):
    """
    Estimates horizontal head rotation (yaw):
    Compares nose tip position relative to left and right cheek boundaries (234, 454).
    Negative = looking right, Positive = looking left (camera perspective).
    """
    nose_x = landmarks[1][0]
    left_cheek_x = landmarks[234][0]
    right_cheek_x = landmarks[454][0]
    cheek_mid_x = (left_cheek_x + right_cheek_x) / 2.0
    yaw = (nose_x - cheek_mid_x) / face_scale
    return yaw

def detect_aggression(prev_landmarks, curr_landmarks):
    """
    Scale-invariant facial aggression and shouting detector.
    Returns:
        aggressive: bool (whether facial expression signifies aggression / quarrel)
        motion_score: float (normalized facial agitation)
        mouth_score: float (normalized MAR, mouth open ratio)
        eyebrow_score: float (eyebrow tension / furrowing score)
        details: dict containing granular metrics
    """
    if curr_landmarks is None:
        return False, 0.0, 0.0, 0.0

    face_scale = compute_face_scale(curr_landmarks)
    mar = mouth_aspect_ratio(curr_landmarks)
    inner_brow, brow_eye = eyebrow_furrow_ratio(curr_landmarks, face_scale)
    rel_motion, head_shift = relative_facial_motion(prev_landmarks, curr_landmarks, face_scale)
    yaw = estimate_head_yaw(curr_landmarks, face_scale)

    # --- NORMALIZED SCORING LOGIC ---
    # Shouting: MAR > 0.35 (strong shouting > 0.48)
    shouting_score = max(0.0, min(1.0, (mar - 0.25) / 0.30))

    # Eyebrow tension: angry frown lowers brow-eye distance (< 0.22)
    # and inner brow distance drops (< 0.65)
    frown_score = 0.0
    if brow_eye < 0.25:
        frown_score += max(0.0, min(1.0, (0.25 - brow_eye) / 0.10)) * 0.6
    if inner_brow < 0.70:
        frown_score += max(0.0, min(1.0, (0.70 - inner_brow) / 0.25)) * 0.4
    frown_score = min(1.0, frown_score)

    # Facial agitation / rapid head movements
    motion_score = min(1.0, rel_motion * 15.0)
    if head_shift > 0.15:  # Rapid head lunge / jerk
        motion_score = min(1.0, motion_score + 0.35)

    # Combined facial aggression confidence (0.0 to 1.0)
    # Shouting + frowning together is a classic argument face
    aggression_confidence = (shouting_score * 0.50) + (frown_score * 0.30) + (motion_score * 0.20)

    # If shouting is very intense, boost confidence
    if mar > 0.48:
        aggression_confidence = max(aggression_confidence, 0.75)

    is_aggressive = aggression_confidence >= 0.45

    # Return standard 4-tuple for backward compatibility + detailed dict
    return is_aggressive, motion_score, mar, frown_score