import math
from collections import deque

class CentroidFaceTracker:
    """
    Tracks multiple faces across frames using centroid matching.
    Prevents face identity swapping and maintains continuous landmark histories.
    """
    def __init__(self, max_distance=0.25, max_lost=10):
        self.next_id = 1
        self.tracks = {}  # track_id -> dict
        self.max_distance = max_distance
        self.max_lost = max_lost

    def update(self, detected_faces):
        """
        detected_faces: list of landmarks:
        Each item is raw landmark list [(x, y), ...] from MediaPipe.
        Returns: list of dicts with keys:
            'id': int,
            'landmarks': list of (x, y),
            'prev_landmarks': list of (x, y) or None,
            'center': (cx, cy),
            'prev_center': (cx, cy),
            'bbox': (min_x, min_y, max_x, max_y),
            'scale': float (face diagonal size for normalization),
            'history': deque of (cx, cy)
        """
        current_detections = []
        for lms in detected_faces:
            xs = [p[0] for p in lms]
            ys = [p[1] for p in lms]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            # Face scale (diagonal) used for scale-invariant normalizations
            scale = math.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
            if scale < 1e-4:
                scale = 1e-4

            current_detections.append({
                'landmarks': lms,
                'center': (cx, cy),
                'bbox': (min_x, min_y, max_x, max_y),
                'scale': scale
            })

        # Match with existing tracks
        updated_tracks = []
        unmatched_detections = list(range(len(current_detections)))
        matched_track_ids = set()

        if self.tracks and current_detections:
            track_ids = list(self.tracks.keys())
            distances = []

            for tid in track_ids:
                tcx, tcy = self.tracks[tid]['center']
                for did, det in enumerate(current_detections):
                    dcx, dcy = det['center']
                    dist = math.sqrt((tcx - dcx) ** 2 + (tcy - dcy) ** 2)
                    distances.append((dist, tid, did))

            # Sort by distance (greedy nearest neighbor matching)
            distances.sort(key=lambda x: x[0])

            for dist, tid, did in distances:
                if dist > self.max_distance:
                    continue
                if tid in matched_track_ids or did not in unmatched_detections:
                    continue

                # Match found
                det = current_detections[did]
                matched_track_ids.add(tid)
                unmatched_detections.remove(did)

                track = self.tracks[tid]
                prev_center = track['center']
                prev_landmarks = track['landmarks']

                track['prev_center'] = prev_center
                track['prev_landmarks'] = prev_landmarks
                track['center'] = det['center']
                track['landmarks'] = det['landmarks']
                track['bbox'] = det['bbox']
                track['scale'] = det['scale']
                track['lost_frames'] = 0
                track['history'].append(det['center'])

                updated_tracks.append({
                    'id': tid,
                    'landmarks': det['landmarks'],
                    'prev_landmarks': prev_landmarks,
                    'center': det['center'],
                    'prev_center': prev_center,
                    'bbox': det['bbox'],
                    'scale': det['scale'],
                    'history': track['history']
                })

        # Register new tracks for unmatched detections
        for did in unmatched_detections:
            det = current_detections[did]
            tid = self.next_id
            self.next_id += 1

            hist = deque(maxlen=30)
            hist.append(det['center'])

            self.tracks[tid] = {
                'id': tid,
                'center': det['center'],
                'prev_center': det['center'],
                'landmarks': det['landmarks'],
                'prev_landmarks': None,
                'bbox': det['bbox'],
                'scale': det['scale'],
                'lost_frames': 0,
                'history': hist
            }

            updated_tracks.append({
                'id': tid,
                'landmarks': det['landmarks'],
                'prev_landmarks': None,
                'center': det['center'],
                'prev_center': det['center'],
                'bbox': det['bbox'],
                'scale': det['scale'],
                'history': hist
            })

        # Increase lost frames for tracks that were not matched
        dead_ids = []
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids and tid not in [t['id'] for t in updated_tracks]:
                self.tracks[tid]['lost_frames'] += 1
                if self.tracks[tid]['lost_frames'] > self.max_lost:
                    dead_ids.append(tid)

        for tid in dead_ids:
            del self.tracks[tid]

        # Sort by ID for deterministic processing
        updated_tracks.sort(key=lambda t: t['id'])
        return updated_tracks
