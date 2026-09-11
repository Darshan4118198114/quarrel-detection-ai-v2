"""
Automated unit and integration test suite for Quarrel Detection AI.
Validates:
1. CentroidFaceTracker ID persistence
2. Scale-invariance of expression_detector metrics
3. QuarrelFusionEngine multi-modal scoring and temporal state transitions
4. InteractionMotionDetector corridor and velocity calculations
5. AlertManager snapshot and logging pipeline
"""

import os
import math
import numpy as np
import cv2

from detection.face_tracker import CentroidFaceTracker
from detection.expression_detector import (
    mouth_aspect_ratio,
    eyebrow_furrow_ratio,
    detect_aggression,
    compute_face_scale
)
from detection.quarrel_detector import QuarrelFusionEngine, check_interaction
from motion.motion_detector import InteractionMotionDetector, calculate_body_motion
from alerts.alert_system import AlertManager
from utils.logger import log_event, get_recent_logs

def generate_mock_face(center=(0.5, 0.5), scale=0.15, mouth_open=0.0):
    """
    Generates 468 mock MediaPipe face landmarks centered at (cx, cy) with given scale.
    """
    cx, cy = center
    lms = []
    for i in range(468):
        angle = (i / 468.0) * 2 * math.pi
        r = scale * (0.8 + 0.2 * math.sin(angle * 3))
        lms.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    # Assign key landmark indices:
    # 33: left outer eye, 263: right outer eye
    lms[33] = (cx - scale * 0.5, cy - scale * 0.2)
    lms[263] = (cx + scale * 0.5, cy - scale * 0.2)
    # 10: forehead, 152: chin
    lms[10] = (cx, cy - scale * 0.9)
    lms[152] = (cx, cy + scale * 0.9)
    # 1: nose tip
    lms[1] = (cx, cy)
    # 234: left cheek, 454: right cheek
    lms[234] = (cx - scale * 0.8, cy)
    lms[454] = (cx + scale * 0.8, cy)
    # Inner lips: 13 (upper), 14 (lower)
    lms[13] = (cx, cy + scale * 0.35 - mouth_open * scale * 0.5)
    lms[14] = (cx, cy + scale * 0.35 + mouth_open * scale * 0.5)
    # Mouth corners: 78 (left), 308 (right)
    lms[78] = (cx - scale * 0.3, cy + scale * 0.35)
    lms[308] = (cx + scale * 0.3, cy + scale * 0.35)
    # Inner eyebrows: 55 (left), 285 (right)
    lms[55] = (cx - scale * 0.2, cy - scale * 0.4)
    lms[285] = (cx + scale * 0.2, cy - scale * 0.4)
    # Upper eyelids: 159 (left), 386 (right)
    lms[159] = (cx - scale * 0.35, cy - scale * 0.22)
    lms[386] = (cx + scale * 0.35, cy - scale * 0.22)
    # Brow reference: 70, 300
    lms[70] = (cx - scale * 0.35, cy - scale * 0.4)
    lms[300] = (cx + scale * 0.35, cy - scale * 0.4)

    return lms

def test_face_tracker():
    print("Testing CentroidFaceTracker...")
    tracker = CentroidFaceTracker(max_distance=0.25)

    # Frame 1: Two faces at (0.3, 0.5) and (0.7, 0.5)
    f1_a = generate_mock_face(center=(0.3, 0.5), scale=0.1)
    f1_b = generate_mock_face(center=(0.7, 0.5), scale=0.1)
    tracks1 = tracker.update([f1_a, f1_b])

    assert len(tracks1) == 2, f"Expected 2 tracks, got {len(tracks1)}"
    id_left = tracks1[0]['id']
    id_right = tracks1[1]['id']

    # Frame 2: Faces moved slightly (0.32, 0.5) and (0.68, 0.5), order passed in reversed
    f2_b = generate_mock_face(center=(0.68, 0.5), scale=0.1)
    f2_a = generate_mock_face(center=(0.32, 0.5), scale=0.1)
    tracks2 = tracker.update([f2_b, f2_a])

    # Check tracks kept their respective persistent IDs
    track_by_id = {t['id']: t for t in tracks2}
    assert id_left in track_by_id, "Left face ID was lost"
    assert id_right in track_by_id, "Right face ID was lost"
    assert track_by_id[id_left]['center'][0] < 0.5, "Left face matched incorrectly"
    assert track_by_id[id_right]['center'][0] > 0.5, "Right face matched incorrectly"
    print("  [PASS] CentroidFaceTracker maintained IDs across reordered detections.")

def test_scale_invariance():
    print("Testing Scale-Invariance of Expression Detector...")
    # Generate same open mouth expression at two drastically different scales (0.08 vs 0.24, 3x zoom)
    face_near = generate_mock_face(center=(0.5, 0.5), scale=0.24, mouth_open=0.5)
    face_far = generate_mock_face(center=(0.5, 0.5), scale=0.08, mouth_open=0.5)

    mar_near = mouth_aspect_ratio(face_near)
    mar_far = mouth_aspect_ratio(face_far)

    # MAR should be identical regardless of distance
    diff = abs(mar_near - mar_far)
    assert diff < 0.05, f"MAR differed significantly across scales: {mar_near:.3f} vs {mar_far:.3f}"

    is_agg_near, _, _, _ = detect_aggression(None, face_near)
    is_agg_far, _, _, _ = detect_aggression(None, face_far)
    assert is_agg_near == is_agg_far, "Aggression detection differed based on scale"
    print(f"  [PASS] Expression detector scale-invariance confirmed (MAR: {mar_near:.3f} vs {mar_far:.3f}).")

def test_quarrel_fusion_engine():
    print("Testing QuarrelFusionEngine...")
    engine = QuarrelFusionEngine(quarrel_duration=0.2, sensitivity=1.0)

    # 1. Calm state (2 faces far apart, mouth closed)
    f_calm_1 = {'id': 1, 'center': (0.15, 0.5), 'bbox': (0.1, 0.4, 0.2, 0.6)}
    f_calm_2 = {'id': 2, 'center': (0.85, 0.5), 'bbox': (0.8, 0.4, 0.9, 0.6)}
    calm_res = engine.evaluate(
        tracked_faces=[f_calm_1, f_calm_2],
        facial_aggressions=[(False, 0.0, 0.1, 0.0), (False, 0.0, 0.1, 0.0)],
        corridor_motion=0.0,
        audio_shouting_score=0.0
    )
    assert calm_res['state'] == engine.STATE_NORMAL, f"Expected NORMAL, got {calm_res['state']}"
    assert calm_res['smoothed_risk'] < 0.25, f"Risk too high for calm: {calm_res['smoothed_risk']}"

    # 2. Confrontation state (close proximity, high facial aggression)
    f_agg_1 = {'id': 1, 'center': (0.42, 0.5), 'bbox': (0.35, 0.4, 0.48, 0.6)}
    f_agg_2 = {'id': 2, 'center': (0.58, 0.5), 'bbox': (0.52, 0.4, 0.65, 0.6)}
    import time
    # Simulate high aggression sustained over time
    for _ in range(10):
        res = engine.evaluate(
            tracked_faces=[f_agg_1, f_agg_2],
            facial_aggressions=[(True, 0.8, 0.55, 0.7), (True, 0.8, 0.55, 0.7)],
            corridor_motion=0.25,
            audio_shouting_score=0.8
        )
        time.sleep(0.03)

    assert res['smoothed_risk'] > 0.70, f"Expected high risk, got {res['smoothed_risk']}"
    assert res['quarrel_confirmed'] is True, "Expected quarrel_confirmed to trigger after duration"
    print("  [PASS] QuarrelFusionEngine correctly transitioned to QUARREL DETECTED.")

def test_motion_detector():
    print("Testing InteractionMotionDetector...")
    mot = InteractionMotionDetector()
    gray1 = np.zeros((240, 320), dtype=np.uint8)
    gray2 = np.zeros((240, 320), dtype=np.uint8)
    # Add motion in center corridor
    gray2[80:160, 100:220] = 200

    mot.update_frame(gray1)
    ratio = mot.compute_corridor_motion(gray2, bbox1=(0.2, 0.3, 0.4, 0.7), bbox2=(0.6, 0.3, 0.8, 0.7))
    assert ratio > 0.1, f"Expected corridor motion, got {ratio}"

    # Test approach velocity
    vel = mot.compute_approach_velocity((0.4, 0.5), (0.6, 0.5), (0.3, 0.5), (0.7, 0.5))
    assert vel > 0, "Approaching subjects should have positive velocity"
    print("  [PASS] InteractionMotionDetector corridor and velocity calculations verified.")

def test_alert_and_logging():
    print("Testing AlertManager & Logger...")
    mgr = AlertManager(incidents_dir="test_incidents", cooldown_seconds=0.1, sound_enabled=False)
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    snap = mgr.trigger_alert(dummy_frame, risk_score=0.92)

    assert snap is not None, "Failed to create snapshot"
    assert os.path.exists(snap), f"Snapshot file not found: {snap}"

    log_event("Automated test log message", severity="TEST", risk_score=0.92, snapshot_path=snap)
    recent = get_recent_logs(5)
    assert any("Automated test log message" in line for line in recent), "Log entry missing from events.log"

    # Cleanup test artifact
    if os.path.exists(snap):
        os.remove(snap)
    if os.path.exists("test_incidents"):
        try:
            os.rmdir("test_incidents")
        except Exception:
            pass
    print("  [PASS] Alert snapshot generation and structured logging verified.")

def run_all_tests():
    print("=" * 60)
    print("Starting Automated Verification of Quarrel Detection AI")
    print("=" * 60)
    test_face_tracker()
    test_scale_invariance()
    test_quarrel_fusion_engine()
    test_motion_detector()
    test_alert_and_logging()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (100% SUCCESS)")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
