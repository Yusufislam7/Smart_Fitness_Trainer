import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# LOAD DATASET
# ========================
csv_path = os.path.join("dataset", "fitness_data.csv")
print("📂 Membaca dataset:", csv_path)

df = pd.read_csv(csv_path)
print("Jumlah data:", len(df))

# Label (gerakan) ada di kolom pertama
labels = df.iloc[:, 0]
features = df.iloc[:, 1:]

print("Fitur shape:", features.shape)
print("Label unik:", labels.unique())

# ========================
# SPLIT DATA
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# ========================
# TRAIN MODEL
# ========================
print("\n🚀 Melatih model Random Forest...")
model = RandomForestClassifier(
    n_estimators=200, random_state=42, max_depth=20
)
model.fit(X_train, y_train)

# ========================
# EVALUASI MODEL
# ========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Akurasi model: {acc * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Fitness Action Recognition")
plt.show()

# ========================
# SIMPAN MODEL
# ========================
model_path = os.path.join("dataset", "fitness_model.pkl")
joblib.dump(model, model_path)
print("💾 Model disimpan ke:", model_path)
