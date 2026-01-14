import os
import cv2
import numpy as np

# =========================
# PATHS
# =========================
RAW_DATASET = "../dataset"
SEG_DATASET = "../segmented_dataset"

IMG_SIZE = 224

# =========================
# SIMPLE BUT EFFECTIVE LEAF SEGMENTATION
# (No deep learning yet — safe first step)
# =========================
def segment_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # green range for leaves
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented

# =========================
# PROCESS DATASET
# =========================
def process_split(split):
    input_root = os.path.join(RAW_DATASET, split)
    output_root = os.path.join(SEG_DATASET, split)

    os.makedirs(output_root, exist_ok=True)

    for disease in os.listdir(input_root):
        in_dir = os.path.join(input_root, disease)
        out_dir = os.path.join(output_root, disease)

        os.makedirs(out_dir, exist_ok=True)

        for img_name in os.listdir(in_dir):
            img_path = os.path.join(in_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            segmented = segment_leaf(img)

            cv2.imwrite(os.path.join(out_dir, img_name), segmented)

if __name__ == "__main__":
    print("Starting segmentation...")
    process_split("train")
    process_split("val")
    print("Segmentation done.")
