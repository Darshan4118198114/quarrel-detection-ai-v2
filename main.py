import cv2
import mediapipe as mp
import time

# Custom modules
from motion.motion_detector import detect_body_aggression
from detection.expression_detector import detect_aggression

# Init MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

prev_faces = []
prev_body = None
start_time = None

print("Starting Quarrel Detection (Body + Expression)...")

while True:
    try:
        ret, frame = cap.read()

        if not ret:
            print("Camera read failed")
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------------- FACE DETECTION ----------------
        results = face_mesh.process(rgb)

        current_faces = []
        aggressive_any = False
        centers = []

        if results.multi_face_landmarks:
            h, w, _ = frame.shape

            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                current_faces.append(landmarks)

                # Face center (for interaction)
                cx = int(face_landmarks.landmark[1].x * w)
                cy = int(face_landmarks.landmark[1].y * h)
                centers.append((cx / w, cy / h))

                # Draw minimal face points (performance)
                for lm in face_landmarks.landmark[::30]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # ---------------- EXPRESSION LOGIC ----------------
        if prev_faces and len(prev_faces) == len(current_faces):
            for prev, curr in zip(prev_faces, current_faces):
                aggressive, motion, mouth, eyebrow = detect_aggression(prev, curr)

                if aggressive:
                    aggressive_any = True

        prev_faces = current_faces

        # ---------------- INTERACTION ----------------
        interaction = False
        if len(centers) >= 2:
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = abs(centers[i][0] - centers[j][0])
                    if dist < 0.25:
                        interaction = True

        # ---------------- BODY MOTION ----------------
        pose_results = pose.process(rgb)

        body_aggressive = False
        body_motion = 0

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark

            # Upper body points
            body_points = [
                (lm[11].x, lm[11].y),  # left shoulder
                (lm[12].x, lm[12].y),  # right shoulder
                (lm[13].x, lm[13].y),  # left elbow
                (lm[14].x, lm[14].y),  # right elbow
                (lm[15].x, lm[15].y),  # left wrist
                (lm[16].x, lm[16].y),  # right wrist
            ]

            body_aggressive, body_motion = detect_body_aggression(prev_body, body_points)
            prev_body = body_points

        # ---------------- DISPLAY ----------------
        cv2.putText(frame, f"Faces: {len(current_faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Body:{body_motion:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if interaction:
            cv2.putText(frame, "Interaction", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        if aggressive_any:
            cv2.putText(frame, "Face Aggression", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

        if body_aggressive and body_motion > 0.015:
            cv2.putText(frame, "Body Aggression", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # ---------------- FINAL DECISION ----------------
        if interaction and (aggressive_any or (body_aggressive and body_motion > 0.015)):
            if start_time is None:
                start_time = time.time()

            elif time.time() - start_time > 2:
                cv2.putText(frame, "QUARREL DETECTED", (50, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        else:
            start_time = None

        # ---------------- SHOW ----------------
        cv2.imshow("Quarrel Detection AI", frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    except Exception as e:
        print("ERROR:", e)
        continue

cap.release()
cv2.destroyAllWindows()