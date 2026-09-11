import math
import cv2
import numpy as np

def calculate_body_motion(prev_points, curr_points, scale=None):
    """
    Computes body motion from landmark points (e.g. MediaPipe Pose).
    If scale is provided (e.g., shoulder width), motion is normalized.
    """
    if prev_points is None or curr_points is None or len(prev_points) != len(curr_points):
        return 0.0

    total_motion = 0.0
    for (x1, y1), (x2, y2) in zip(prev_points, curr_points):
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_motion += dist

    avg_motion = total_motion / len(curr_points)
    if scale and scale > 1e-4:
        return avg_motion / scale
    return avg_motion


def detect_body_aggression(prev_points, curr_points, threshold=0.08):
    """
    Backward-compatible body aggression check.
    """
    if prev_points is None:
        return False, 0.0

    motion = calculate_body_motion(prev_points, curr_points)
    aggressive = motion > threshold
    return aggressive, motion


class InteractionMotionDetector:
    """
    Analyzes physical motion dynamics between multiple interacting subjects:
    1. Rapid approach / lunging velocity (closing the gap between subjects)
    2. Dynamic motion energy in the confrontation corridor between subjects
    3. Per-person upper-body agitation
    """
    def __init__(self, history_len=5):
        self.prev_gray = None
        self.prev_centers = {}

    def compute_approach_velocity(self, p1_center, p2_center, prev_p1_center, prev_p2_center):
        """
        Computes how fast two subjects are closing distance toward each other.
        Positive value = moving closer (lunging/confronting).
        Negative value = moving apart.
        """
        if prev_p1_center is None or prev_p2_center is None:
            return 0.0

        curr_dist = math.sqrt((p1_center[0] - p2_center[0])**2 + (p1_center[1] - p2_center[1])**2)
        prev_dist = math.sqrt((prev_p1_center[0] - prev_p2_center[0])**2 + (prev_p1_center[1] - prev_p2_center[1])**2)

        # Delta distance (positive when approaching)
        approach_rate = prev_dist - curr_dist
        return approach_rate

    def compute_corridor_motion(self, frame_gray, bbox1, bbox2):
        """
        Measures motion energy in the spatial corridor between two subjects.
        In fights, shoves, or heated arguments, this zone exhibits high arm/hand motion.
        """
        if self.prev_gray is None:
            self.prev_gray = frame_gray.copy()
            return 0.0

        h, w = frame_gray.shape

        # Corridor between the two bounding boxes
        min_x = int(min(bbox1[0], bbox2[0]) * w)
        max_x = int(max(bbox1[2], bbox2[2]) * w)
        min_y = int(min(bbox1[1], bbox2[1]) * h)
        max_y = int(max(bbox1[3], bbox2[3]) * h)

        # Clamp to frame bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(w, max_x)
        max_y = min(h, max_y)

        if (max_x - min_x) < 10 or (max_y - min_y) < 10:
            return 0.0

        diff = cv2.absdiff(frame_gray[min_y:max_y, min_x:max_x],
                           self.prev_gray[min_y:max_y, min_x:max_x])
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_ratio = np.count_nonzero(thresh) / float(thresh.size)

        return motion_ratio

    def update_frame(self, frame_gray):
        self.prev_gray = frame_gray.copy()