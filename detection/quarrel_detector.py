import time
from utils.helpers import calculate_distance

start_time = None

def detect_quarrel(motion_score, centers):
    global start_time

    if len(centers) >= 2:
        dist = calculate_distance(centers[0], centers[1])

        if motion_score > 0.6 and dist < 200:
            if start_time is None:
                start_time = time.time()

            if time.time() - start_time > 2:
                return True
        else:
            start_time = None

    return False