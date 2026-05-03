import math

def calculate_body_motion(prev_points, curr_points):
    if prev_points is None:
        return 0

    total_motion = 0

    for (x1, y1), (x2, y2) in zip(prev_points, curr_points):
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_motion += dist

    return total_motion / len(curr_points)


def detect_body_aggression(prev_points, curr_points):
    if prev_points is None:
        return False, 0

    motion = calculate_body_motion(prev_points, curr_points)

    # 🔥 stronger logic
    aggressive = motion > 0.02   # increase sensitivity

    return aggressive, motion