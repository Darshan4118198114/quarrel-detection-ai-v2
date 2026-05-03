import math

def face_motion(prev_landmarks, curr_landmarks):
    motion = 0
    for (x1, y1), (x2, y2) in zip(prev_landmarks, curr_landmarks):
        motion += abs(x2 - x1) + abs(y2 - y1)
    return motion / len(curr_landmarks)

def mouth_open_score(landmarks):
    # approximate mouth points
    top = landmarks[13]
    bottom = landmarks[14]
    return abs(bottom[1] - top[1])

def eyebrow_raise_score(landmarks):
    left_eye = landmarks[159]
    left_brow = landmarks[65]
    return abs(left_eye[1] - left_brow[1])

def detect_aggression(prev_landmarks, curr_landmarks):
    if prev_landmarks is None:
        return False, 0, 0, 0, 0

    motion = face_motion(prev_landmarks, curr_landmarks)
    mouth = mouth_open_score(curr_landmarks)
    eyebrow = eyebrow_raise_score(curr_landmarks)

    # 🔥 tension (micro movement intensity)
    tension = 0
    for (x1, y1), (x2, y2) in zip(prev_landmarks, curr_landmarks):
        tension += abs(x2 - x1) + abs(y2 - y1)
    tension /= len(curr_landmarks)

    # weighted score
    score = (
        motion * 0.4 +
        mouth * 0.2 +
        eyebrow * 0.2 +
        tension * 0.2
    )

    aggressive = score > 0.025

    return aggressive, motion, mouth, eyebrow,