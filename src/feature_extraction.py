import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern, hog
import pickle

FACES_FOLDER = "DATASET/FACES"

features = []
labels = []

# 🔥 counters for balancing
real_count = 0
morph_count = 0
LIMIT = 20   # you can change (20, 30, etc.)

for filename in os.listdir(FACES_FOLDER):
    print("Processing:", filename)

    path = os.path.join(FACES_FOLDER, filename)

    img = cv2.imread(path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (300, 300))

    # -------- LBP --------
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256)

    # -------- HOG --------
    hog_features = hog(
        gray,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        feature_vector=True
    )

    combined = np.hstack((lbp_hist, hog_features))

    # -------- LABELING + BALANCING --------
    if "real" in filename.lower():
        if real_count >= LIMIT:
            continue
        label = 0
        real_count += 1

    elif "morph" in filename.lower():
        if morph_count >= LIMIT:
            continue
        label = 1
        morph_count += 1

    else:
        print("Skipped:", filename)
        continue

    features.append(combined)
    labels.append(label)

# -------------------------
# DEBUG OUTPUT
# -------------------------
print("\nLabels present:", set(labels))
print("Real:", labels.count(0), "Morph:", labels.count(1))

# -------------------------
# SAVE FEATURES
# -------------------------
data = {"features": features, "labels": labels}

with open("features.pkl", "wb") as f:
    pickle.dump(data, f)

print("\n✅ Feature extraction completed and saved!")