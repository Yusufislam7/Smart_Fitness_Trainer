import cv2
import mediapipe as mp
import numpy as np
import joblib

# === Load Model ===
model = joblib.load('dataset/fitness_model.pkl')

# === MediaPipe Pose Setup ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# === OpenCV Webcam ===
cap = cv2.VideoCapture(0)

actions = ['pushup', 'squat', 'situp']  # sesuaikan dengan label model kamu

current_action = None
previous_action = None
reps = 0
cooldown = 0  # mencegah double count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Ambil koordinat (x, y, z)
        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)
            landmarks.append(lm.z)

        landmarks = np.array(landmarks).reshape(1, -1)

        # Prediksi aksi
        prediction = model.predict(landmarks)[0]
        proba = model.predict_proba(landmarks)[0].max() * 100

        # === Logic Repetition Counter ===
        if proba > 85:  # hanya hitung jika yakin
            current_action = prediction
            if previous_action != current_action and cooldown == 0:
                # misalnya: dari "down" ke "up"
                reps += 1
                cooldown = 10  # jeda 10 frame biar gak double count
            previous_action = current_action
        else:
            current_action = None

        if cooldown > 0:
            cooldown -= 1

        # === Overlay UI ===
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # === Teks & Style ===
        text = f"Action: {prediction.upper()} | Confidence: {proba:.1f}%"
        cv2.putText(frame, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

        # === Confidence Bar ===
        bar_x, bar_y = 20, 70
        bar_length = int((proba / 100) * 300)
        cv2.rectangle(frame, (bar_x, bar_y + 10), (bar_x + 300, bar_y + 25), (80, 80, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y + 10), (bar_x + bar_length, bar_y + 25), (255, 255, 255), -1)

        # === Reps Counter ===
        cv2.putText(frame, f"REPS: {reps}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow("💪 Smart Fitness Trainer (Auto Counter)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
