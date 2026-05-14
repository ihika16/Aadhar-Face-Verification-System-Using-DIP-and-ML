import cv2
import os
import random

# Folder containing real faces
INPUT_FOLDER = "DATASET/FACES"

# Folder to save morph images
OUTPUT_FOLDER = "DATASET/MORPH"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

images = os.listdir(INPUT_FOLDER)

count = 0

# 🔥 Number of morph images to create
NUM_MORPHS = 20

for i in range(NUM_MORPHS):
    # pick 2 random images
    img1_name, img2_name = random.sample(images, 2)

    img1_path = os.path.join(INPUT_FOLDER, img1_name)
    img2_path = os.path.join(INPUT_FOLDER, img2_name)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        continue

    # resize both images
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    # 🔥 create morph (blend)
    morph = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    # save image
    save_path = os.path.join(OUTPUT_FOLDER, f"morph_{count}.jpg")
    cv2.imwrite(save_path, morph)

    count += 1

print("✅ Morph images created:", count)