import cv2
import mediapipe as mp

from motion.motion_detector import calculate_motion
from detection.quarrel_detector import detect_quarrel
from alerts.alert_system import send_alert

print("Starting Quarrel Detection (Modular)...")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

prev_landmarks = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    centers = []

    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2
        centers.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

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

            cv2.putText(frame, f"Motion: {motion_score:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if detect_quarrel(motion_score, centers):
                cv2.putText(frame, "QUARREL DETECTED", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                send_alert()

        prev_landmarks = landmarks

    cv2.imshow("Quarrel Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()