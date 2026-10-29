import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# ========================
# CONFIGURASI DATASET
# ========================
DATA_DIR = "dataset"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Nama file CSV untuk menyimpan data
csv_path = os.path.join(DATA_DIR, "fitness_data.csv")

# Kelas gerakan yang ingin direkam (ubah sesuai kebutuhan)
actions = ["squat", "pushup", "situp", "jumpingjack", "stand"]

# Jumlah frame yang ingin diambil per aksi
frames_per_action = 200

# ========================
# INISIALISASI MEDIAPIPE
# ========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ========================
# FUNGSI SIMPAN KE CSV
# ========================
def save_landmarks_to_csv(label, landmarks):
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        row = [label]
        for lm in landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
        writer.writerow(row)

# ========================
# MULAI PENGAMBILAN DATA
# ========================
cap = cv2.VideoCapture(0)
print("📸 Kamera siap...")

for action in actions:
    print(f"\n=== Sekarang ambil data untuk gerakan: {action.upper()} ===")
    input("Tekan ENTER jika kamu siap...")

    count = 0
    while count < frames_per_action:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Gagal membaca frame kamera!")
            break

        # Flip biar mirror view
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            save_landmarks_to_csv(action, results.pose_landmarks)
            count += 1
            cv2.putText(frame, f"Recording {action}: {count}/{frames_per_action}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("\n✅ Pengambilan data selesai!")
cap.release()
cv2.destroyAllWindows()
