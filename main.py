import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

print("Starting Quarrel Detection...")

# Load YOLO model
model = YOLO("yolov8n.pt")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

prev_landmarks = None
start_time = None

def calculate_motion(prev, curr):
    motion = 0
    for p, c in zip(prev, curr):
        motion += np.linalg.norm(np.array(p) - np.array(c))
    return motion

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO person detection
    results = model(frame)

    people_centers = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                people_centers.append((cx, cy))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

    # Pose detection (single for now)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = [(lm.x, lm.y) for lm in result.pose_landmarks.landmark]

        if prev_landmarks:
            motion_score = calculate_motion(prev_landmarks, landmarks)

            cv2.putText(frame, f"Motion: {motion_score:.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # 🔥 Interaction logic
            if len(people_centers) >= 2:
                dist = calculate_distance(people_centers[0], people_centers[1])

                if motion_score > 0.6 and dist < 150:
                    if start_time is None:
                        start_time = time.time()

                    if time.time() - start_time > 2:
                        cv2.putText(frame, "⚠️ QUARREL DETECTED", (50,80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    start_time = None

        prev_landmarks = landmarks

    cv2.imshow("Quarrel Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()