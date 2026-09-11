import argparse
import os
import sys
import time
import math
import cv2
import mediapipe as mp
import numpy as np

from detection.face_tracker import CentroidFaceTracker
from detection.expression_detector import detect_aggression
from detection.quarrel_detector import QuarrelFusionEngine
from motion.motion_detector import InteractionMotionDetector
from audio.audio_detector import AudioShoutingDetector
from alerts.alert_system import send_alert, AlertManager
from utils.logger import log_event
from utils.visualizer import HUDVisualizer

def get_video_capture(source_str):
    """
    Resolves video capture source from:
    - Camera index (0, 1, etc.)
    - IP Webcam or DroidCam URL (e.g. http://192.168.x.x:8080/video)
    - Video file path (e.g. sample.mp4)
    - Synthetic test generator ("test" or "demo")
    """
    if source_str is None or source_str.strip() == "":
        source_str = os.environ.get("CAMERA_SOURCE", "0")

    source_str = source_str.strip()

    # Synthetic test mode
    if source_str.lower() in ("test", "demo", "synth"):
        return None, "Synthetic Test Mode"

    # Numeric camera index
    if source_str.isdigit():
        idx = int(source_str)
        # Try DirectShow on Windows for fast init
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        return cap, f"Camera Index {idx}"

    # Handle IP / URL format
    url = source_str
    if not (url.startswith("http://") or url.startswith("https://") or url.startswith("rtsp://")):
        if ":" in url and not os.path.exists(url):
            url = f"http://{url}"

    # Auto-complete common mobile camera endpoints if omitted
    if url.startswith("http://") or url.startswith("https://"):
        if ":8080" in url and not url.endswith(("/video", "/shot.jpg", "/videofeed")):
            url = url.rstrip("/") + "/video"
        elif ":4747" in url and not url.endswith(("/video", "/mjpegfeed")):
            url = url.rstrip("/") + "/video"

    cap = cv2.VideoCapture(url)
    return cap, url

def print_connection_help(source_name):
    print(f"\n[ERROR] Could not open video source: '{source_name}'")
    print("=" * 65)
    print("How to use a Mobile Camera or Video with this model:")
    print(" Option 1: IP Webcam App (Android)")
    print("   1. Install 'IP Webcam' from Google Play Store on your phone.")
    print("   2. Connect phone & PC to the same Wi-Fi network.")
    print("   3. Start server in app and note IP (e.g. http://192.168.1.15:8080).")
    print("   4. Run: python main.py --source http://<YOUR_PHONE_IP>:8080/video")
    print("\n Option 2: DroidCam App (Android / iOS)")
    print("   1. Open DroidCam on phone.")
    print("   2. Run: python main.py --source http://<YOUR_PHONE_IP>:4747/video")
    print("\n Option 3: Test with a Video File")
    print("   Run: python main.py --source path/to/video.mp4")
    print("\n Option 4: Synthetic Demo Test (No camera required)")
    print("   Run: python main.py --source test")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Quarrel Detection AI - Advanced Multi-Modal Surveillance")
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Video source: camera index (0, 1), IP camera URL (http://<ip>:8080/video), video file, or 'test'."
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable microphone audio shouting detection."
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Detection sensitivity multiplier (default: 1.0, lower=fewer alerts, higher=more sensitive)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (useful for servers / background services)."
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Path to record output video with HUD overlays (e.g., output.mp4)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop processing after N frames (useful for automated benchmarks)."
    )
    args = parser.parse_args()

    # Initialize video capture
    is_test_mode = (args.source and args.source.strip().lower() in ("test", "demo", "synth"))
    if not is_test_mode:
        cap, source_name = get_video_capture(args.source)
        if not cap.isOpened():
            print_connection_help(source_name)
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = None
        source_name = "Synthetic Test Simulator"

    # Initialize AI Detectors & Modules
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    face_tracker = CentroidFaceTracker(max_distance=0.25, max_lost=15)
    motion_detector = InteractionMotionDetector()
    fusion_engine = QuarrelFusionEngine(sensitivity=args.sensitivity)
    audio_detector = AudioShoutingDetector(enabled=(not args.no_audio))
    alert_mgr = AlertManager()
    hud = HUDVisualizer()

    # Video recording setup if requested
    video_writer = None

    print(f"\n=======================================================")
    print(f"[START] Quarrel Detection AI Initialized")
    print(f"[SOURCE] Video Source: {source_name}")
    print(f"[AUDIO] Audio Shouting Detection: {'ACTIVE' if not args.no_audio else 'DISABLED'}")
    print(f"[CONFIG] Sensitivity: {args.sensitivity}x")
    print(f"[INPUT] Press 'q' or 'ESC' to exit, 's' for manual snapshot.")
    print(f"=======================================================\n")

    frame_count = 0
    fail_count = 0
    fps = 30.0
    last_time = time.time()
    alert_active = False

    try:
        while True:
            if is_test_mode:
                # Generate synthetic test frame with two animated figures
                frame_count += 1
                t = frame_count * 0.05
                frame = np.full((480, 640, 3), 35, dtype=np.uint8)

                # Simulate two approaching heads during argument
                dist_sep = max(0.18, 0.40 - 0.15 * math.sin(t * 0.5))
                c1_x, c1_y = 0.5 - dist_sep / 2, 0.5 + 0.05 * math.sin(t * 3.0)
                c2_x, c2_y = 0.5 + dist_sep / 2, 0.5 - 0.05 * math.sin(t * 3.0)

                # Draw synthetic subjects
                for cx, cy, col in [(c1_x, c1_y, (200, 150, 50)), (c2_x, c2_y, (50, 150, 200))]:
                    px, py = int(cx * 640), int(cy * 480)
                    cv2.circle(frame, (px, py), 45, col, -1)
                    cv2.rectangle(frame, (px - 55, py + 45), (px + 55, py + 180), col, -1)

                time.sleep(0.033)
            else:
                ret, frame = cap.read()
                if not ret:
                    fail_count += 1
                    if fail_count > 60:
                        print("\n⚠️ Video stream ended or disconnected.")
                        break
                    time.sleep(0.03)
                    continue
                fail_count = 0

                # Flip horizontally if using default webcam for natural mirror view
                if args.source is None or args.source.isdigit():
                    frame = cv2.flip(frame, 1)

            frame_count += 1
            h, w = frame.shape[:2]

            # Calculate live FPS
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Resize for MediaPipe face detection speed
            small_rgb = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB)
            small_gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)

            # 1. Face Mesh Detection
            results = face_mesh.process(small_rgb)
            detected_face_landmarks = []
            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    # Normalized landmarks (0.0 to 1.0)
                    landmarks = [(lm.x, lm.y) for lm in face_lms.landmark]
                    detected_face_landmarks.append(landmarks)

            # 2. Multi-Person Identity Tracking
            tracked_faces = face_tracker.update(detected_face_landmarks)

            # 3. Facial Expression & Shouting Analysis per Subject
            facial_aggressions = []
            for face in tracked_faces:
                is_agg, mot_score, mar, frown_score = detect_aggression(
                    face['prev_landmarks'],
                    face['landmarks']
                )
                facial_aggressions.append((is_agg, mot_score, mar, frown_score))

            # 4. Motion Dynamics in Interaction Corridor
            corridor_motion = 0.0
            if len(tracked_faces) >= 2:
                corridor_motion = motion_detector.compute_corridor_motion(
                    small_gray,
                    tracked_faces[0]['bbox'],
                    tracked_faces[1]['bbox']
                )
            motion_detector.update_frame(small_gray)

            # 5. Audio Agitation / Shouting Query
            audio_status = audio_detector.get_status()
            audio_shouting_score = audio_status[1] if audio_status[3] else 0.0

            # 6. Multi-Modal Fusion Evaluation
            eval_result = fusion_engine.evaluate(
                tracked_faces=tracked_faces,
                facial_aggressions=facial_aggressions,
                corridor_motion=corridor_motion,
                audio_shouting_score=audio_shouting_score
            )

            # 7. Alert & Snapshot Triggering
            if eval_result['quarrel_confirmed']:
                if not alert_active:
                    snapshot_path = alert_mgr.trigger_alert(
                        frame=frame,
                        risk_score=eval_result['smoothed_risk'],
                        details=eval_result['details']
                    )
                    log_event(
                        "Quarrel / Physical Confrontation Confirmed",
                        severity="ALERT",
                        risk_score=eval_result['smoothed_risk'],
                        snapshot_path=snapshot_path
                    )
                    alert_active = True
            else:
                alert_active = False

            # 8. Render Modern Surveillance HUD
            display_frame = frame.copy()
            display_frame = hud.draw_hud(
                display_frame,
                eval_result=eval_result,
                tracked_faces=tracked_faces,
                fps=fps,
                audio_status=audio_status,
                source_name=source_name
            )

            # Record frame if requested
            if args.record:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(args.record, fourcc, 25.0, (w, h))
                video_writer.write(display_frame)

            # 9. Display Output
            if not args.headless:
                cv2.imshow("Quarrel Detection AI - Surveillance HUD", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("\n[EXIT] Exiting Quarrel Detection AI.")
                    break
                elif key == ord('s'):
                    # Manual snapshot
                    snap_path = alert_mgr.trigger_alert(display_frame, risk_score=eval_result['smoothed_risk'])
                    print(f"[SNAPSHOT] Manual snapshot saved: {snap_path}")

            if args.max_frames and frame_count >= args.max_frames:
                print(f"\nReached max frames ({args.max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Interrupted by user.")
    finally:
        # Cleanup
        audio_detector.stop()
        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete.")

if __name__ == "__main__":
    main()