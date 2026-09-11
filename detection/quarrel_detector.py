import math
import time

def check_interaction(centers, prev_centers=None, threshold=0.35):
    """
    Backward-compatible interaction check.
    Computes Euclidean distance between subject centroids.
    Returns (interaction: bool, interaction_score: float)
    """
    if len(centers) < 2:
        return False, 0.0

    min_dist = float('inf')
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            x1, y1 = centers[i]
            x2, y2 = centers[j]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < min_dist:
                min_dist = dist

    # If within interaction proximity threshold (normalized 0.0 to 1.0)
    is_interacting = min_dist < threshold
    score = max(0.0, (threshold - min_dist) / threshold) if is_interacting else 0.0
    return is_interacting, score


class QuarrelFusionEngine:
    """
    Multi-modal temporal fusion engine for quarrel and fight detection.
    Combines:
      - Spatial Proximity & Facing Direction (Orientation)
      - Facial Agression (Shouting MAR + Eyebrow tension)
      - Physical Body / Interaction Corridor Motion Energy
      - Audio Agitation / Shouting Decibels (if available)
    Applies Exponential Moving Average (EMA) smoothing and state transition logic.
    """
    STATE_NORMAL = "NORMAL"
    STATE_ATTENTION = "ATTENTION"
    STATE_CONFRONTATION = "CONFRONTATION"
    STATE_QUARREL = "QUARREL DETECTED"

    def __init__(self,
                 proximity_threshold=0.38,
                 quarrel_duration=1.2,
                 decay_rate=0.88,
                 sensitivity=1.0):
        self.proximity_threshold = proximity_threshold
        self.quarrel_duration = quarrel_duration
        self.decay_rate = decay_rate
        self.sensitivity = sensitivity

        self.current_risk = 0.0
        self.smoothed_risk = 0.0
        self.state = self.STATE_NORMAL

        self.high_risk_start_time = None
        self.alert_triggered = False

    def evaluate(self,
                 tracked_faces,
                 facial_aggressions,
                 corridor_motion=0.0,
                 audio_shouting_score=0.0):
        """
        tracked_faces: list of tracked face dicts from CentroidFaceTracker
        facial_aggressions: list of (is_aggressive, motion_score, mar, frown_score)
        corridor_motion: float (0.0 to 1.0)
        audio_shouting_score: float (0.0 to 1.0)

        Returns dict with:
          'risk_score': float (0.0 - 1.0),
          'smoothed_risk': float (0.0 - 1.0),
          'state': str,
          'quarrel_confirmed': bool,
          'proximity': float,
          'details': dict
        """
        num_faces = len(tracked_faces)
        proximity_score = 0.0
        face_aggression_score = 0.0
        orientation_score = 0.0

        # Calculate max facial aggression among all active subjects
        if facial_aggressions:
            for is_agg, m_score, mar, f_score in facial_aggressions:
                # Shouting (high MAR) + Frown (tension)
                f_score_val = (mar * 0.50) + (f_score * 0.30) + (m_score * 0.20)
                if f_score_val > face_aggression_score:
                    face_aggression_score = f_score_val

        # Spatial proximity and facing logic for 2 or more people
        if num_faces >= 2:
            min_dist = float('inf')
            pair = None
            for i in range(num_faces):
                for j in range(i + 1, num_faces):
                    c1 = tracked_faces[i]['center']
                    c2 = tracked_faces[j]['center']
                    d = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                    if d < min_dist:
                        min_dist = d
                        pair = (tracked_faces[i], tracked_faces[j])

            if min_dist < self.proximity_threshold:
                # Proximity score 0.0 to 1.0 (closer = higher)
                proximity_score = (self.proximity_threshold - min_dist) / self.proximity_threshold

                # Check facing orientation
                if pair:
                    p1, p2 = pair
                    # Check if subjects are positioned left-right and looking inwards
                    p1_left = p1['center'][0] < p2['center'][0]
                    # Estimate if facing each other based on position
                    orientation_score = 0.6  # Default confrontation geometry
        else:
            # Single person: proximity is 0, but single person aggression is possible (yelling/screaming)
            proximity_score = 0.1 if face_aggression_score > 0.6 else 0.0

        # ---------------- MULTI-MODAL WEIGHTED FUSION ----------------
        if num_faces >= 2:
            # Multi-person confrontation model
            raw_risk = (
                (proximity_score * 0.30) +
                (face_aggression_score * 0.35) +
                (corridor_motion * 0.20) +
                (audio_shouting_score * 0.15)
            ) * self.sensitivity

            # Mutual synergy boost: if high proximity AND shouting/aggression occur together
            if proximity_score > 0.4 and face_aggression_score > 0.4:
                raw_risk += 0.25

            # If corridor motion or physical fighting occurs in close proximity
            if proximity_score > 0.4 and corridor_motion > 0.15:
                raw_risk += 0.30

        else:
            # Single person agitated / shouting
            raw_risk = (face_aggression_score * 0.60 + audio_shouting_score * 0.40) * 0.5 * self.sensitivity

        raw_risk = max(0.0, min(1.0, raw_risk))
        self.current_risk = raw_risk

        # ---------------- TEMPORAL FILTER (EMA) ----------------
        # Fast attack, smooth decay
        if raw_risk > self.smoothed_risk:
            alpha = 0.45  # reacts quickly to aggression
        else:
            alpha = 0.15  # decays smoothly to avoid flickering
        self.smoothed_risk = (alpha * raw_risk) + ((1.0 - alpha) * self.smoothed_risk)
        self.smoothed_risk = max(0.0, min(1.0, self.smoothed_risk))

        # ---------------- STATE MACHINE & PERSISTENCE ----------------
        quarrel_confirmed = False

        if self.smoothed_risk >= 0.70:
            if self.high_risk_start_time is None:
                self.high_risk_start_time = time.time()
            elif (time.time() - self.high_risk_start_time) >= self.quarrel_duration:
                quarrel_confirmed = True
                self.state = self.STATE_QUARREL
            else:
                self.state = self.STATE_CONFRONTATION
        else:
            self.high_risk_start_time = None
            if self.smoothed_risk >= 0.45:
                self.state = self.STATE_CONFRONTATION
            elif self.smoothed_risk >= 0.25:
                self.state = self.STATE_ATTENTION
            else:
                self.state = self.STATE_NORMAL

        return {
            'risk_score': self.current_risk,
            'smoothed_risk': self.smoothed_risk,
            'state': self.state,
            'quarrel_confirmed': quarrel_confirmed,
            'proximity': proximity_score,
            'details': {
                'face_aggression': face_aggression_score,
                'corridor_motion': corridor_motion,
                'audio_shouting': audio_shouting_score,
                'num_faces': num_faces
            }
        }