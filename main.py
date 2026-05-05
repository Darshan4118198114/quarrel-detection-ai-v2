import cv2
import mediapipe as mp
import time

from motion.motion_detector import detect_body_aggression
from detection.expression_detector import detect_aggression

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=False
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# Reduce resolution (fix lag)
cap.set(3, 640)
cap.set(4, 480)

prev_faces = []
prev_body = None

expression_start = None
final_start = None

frame_count = 0

print("Starting Quarrel Detection (DEBUG MODE)...")

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % 2 != 0:
            continue  # skip frames for performance

        frame = cv2.flip(frame, 1)

        # smaller frame for processing
        small = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ---------------- FACE ----------------
        results = face_mesh.process(rgb)

        current_faces = []
        centers = []
        aggressive_any = False

        motion = mouth = eyebrow = 0  # debug default

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                current_faces.append(landmarks)

                cx = face_landmarks.landmark[1].x
                cy = face_landmarks.landmark[1].y
                centers.append((cx, cy))

        # ---------------- EXPRESSION ----------------
        if prev_faces and len(prev_faces) == len(current_faces):
            for prev, curr in zip(prev_faces, current_faces):
                aggressive, motion, mouth, eyebrow = detect_aggression(prev, curr)

                # 🔥 PRINT DEBUG (terminal)
                print(f"motion:{motion:.4f} | mouth:{mouth:.4f} | eyebrow:{eyebrow:.4f}")

                if aggressive:
                    aggressive_any = True

        prev_faces = current_faces

        # ---------------- TEMPORAL FILTER ----------------
        confirmed_expression = False

        if aggressive_any:
            if expression_start is None:
                expression_start = time.time()
            elif time.time() - expression_start > 0.5:
                confirmed_expression = True
        else:
            expression_start = None

        # ---------------- INTERACTION ----------------
        interaction = False
        if len(centers) >= 2:
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = abs(centers[i][0] - centers[j][0])
                    if dist < 0.25:
                        interaction = True

        # ---------------- BODY ----------------
        pose_results = pose.process(rgb)

        body_aggressive = False
        body_motion = 0

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark

            body_points = [
                (lm[11].x, lm[11].y),
                (lm[12].x, lm[12].y),
                (lm[13].x, lm[13].y),
                (lm[14].x, lm[14].y),
                (lm[15].x, lm[15].y),
                (lm[16].x, lm[16].y),
            ]

            body_aggressive, body_motion = detect_body_aggression(prev_body, body_points)
            prev_body = body_points

        # ---------------- DISPLAY ----------------
        cv2.putText(frame, f"Faces: {len(current_faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Body:{body_motion:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # 🔥 DEBUG ON SCREEN
        cv2.putText(frame, f"M:{motion:.3f}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(frame, f"Mo:{mouth:.3f}", (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(frame, f"E:{eyebrow:.3f}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        if interaction:
            cv2.putText(frame, "Interaction", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        if confirmed_expression:
            cv2.putText(frame, "Face Aggression", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

        if body_aggressive:
            cv2.putText(frame, "Body Aggression", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # ---------------- FINAL ----------------
        quarrel = interaction and (confirmed_expression or body_aggressive)

        if quarrel:
            if final_start is None:
                final_start = time.time()
            elif time.time() - final_start > 1.5:
                cv2.putText(frame, "QUARREL DETECTED", (50, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        else:
            final_start = None

        cv2.imshow("Quarrel Detection AI", frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    except Exception as e:
        print("ERROR:", e)
        continue

cap.release()
cv2.destroyAllWindows()