import cv2
import os

INPUT_FOLDER = "DATASET/AADHAR_IMAGES"
OUTPUT_FOLDER = "DATASET/FACES"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

count = 0

for subfolder in os.listdir(INPUT_FOLDER):
    subfolder_path = os.path.join(INPUT_FOLDER, subfolder)

    if not os.path.isdir(subfolder_path):
        continue

    for filename in os.listdir(subfolder_path):
        print("Processing:", filename)

        img_path = os.path.join(subfolder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print("Skipped:", filename)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (300, 300))

            save_path = os.path.join(OUTPUT_FOLDER, f"face_{count}.jpg")
            cv2.imwrite(save_path, face)

            count += 1

print("✅ Faces extracted:", count)