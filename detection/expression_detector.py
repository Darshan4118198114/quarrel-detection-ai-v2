import math


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def face_motion(prev, curr):
    motion = 0
    for (x1, y1), (x2, y2) in zip(prev, curr):
        motion += abs(x2 - x1) + abs(y2 - y1)
    return motion / len(curr)


def mouth_open_score(landmarks):
    return abs(landmarks[13][1] - landmarks[14][1])


def eyebrow_score(landmarks):
    return abs(landmarks[65][1] - landmarks[159][1])


def detect_aggression(prev_landmarks, curr_landmarks):
    if prev_landmarks is None:
        return False, 0, 0, 0

    motion = face_motion(prev_landmarks, curr_landmarks)
    mouth = mouth_open_score(curr_landmarks)
    eyebrow = eyebrow_score(curr_landmarks)

    # ----------- SCORING SYSTEM -----------

    score = 0

    # motion = strongest indicator
    if motion > 0.01:
        score += 2

    # mouth (shouting)
    if mouth > 0.015:
        score += 1

    # eyebrow tension
    if eyebrow > 0.008:
        score += 1

    # extra boost for strong motion
    if motion > 0.02:
        score += 2

    # ----------- FINAL DECISION -----------

    aggressive = score >= 2

    return aggressive, motion, mouth, eyebrow