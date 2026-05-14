import cv2
import os
import numpy as np
import pickle
from skimage.feature import local_binary_pattern, hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# -----------------------------
# STEP 1: Load saved features
# -----------------------------
with open("features.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["features"])
y = np.array(data["labels"])

print("Training samples:", len(X))

# -----------------------------
# STEP 2: SCALE FEATURES (VERY IMPORTANT)
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# STEP 3: Train IMPROVED MODEL
# -----------------------------
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
model.fit(X, y)

print("✅ Model trained successfully")

# -----------------------------
# FEATURE EXTRACTION FUNCTION
# -----------------------------
def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (300, 300))

    # LBP
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256)

    # Normalize histogram
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # HOG
    hog_features = hog(gray,
                       pixels_per_cell=(16,16),
                       cells_per_block=(2,2),
                       feature_vector=True)

    return np.hstack((lbp_hist, hog_features))

# -----------------------------
# STEP 4: TEST MULTIPLE IMAGES
# -----------------------------
TEST_IMAGES = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]

print("\n🎯 RESULTS:\n")

for file in TEST_IMAGES:

    if not os.path.exists(file):
        print(f"{file} → ❌ Not Found")
        continue

    img = cv2.imread(file)

    if img is None:
        print(f"{file} → ❌ Error loading")
        continue

    features = extract_features(img)

    # Scale test features
    features = scaler.transform([features])

    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]

    if prediction == 0:
        print(f"{file} → ✅ REAL IMAGE (Confidence: {confidence[0]:.2f})")
    else:
        print(f"{file} → ❌ MORPH IMAGE (Confidence: {confidence[1]:.2f})")