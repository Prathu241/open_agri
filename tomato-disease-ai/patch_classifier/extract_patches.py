import os
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# PATHS
# =========================
SEG_DATASET = "../segmented_dataset_v2"
PATCH_DATASET = "../patch_dataset"

PATCH_SIZE = 96
PATCHES_PER_IMAGE = 32
MIN_LEAF_RATIO = 0.3
   # % of non-black pixels

# =========================
# PATCH EXTRACTION
# =========================
def extract_patches(image):
    h, w, _ = image.shape
    patches = []

    for _ in range(PATCHES_PER_IMAGE * 3):  # try more, keep best
        if len(patches) >= PATCHES_PER_IMAGE:
            break

        x = np.random.randint(0, w - PATCH_SIZE)
        y = np.random.randint(0, h - PATCH_SIZE)

        patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        # Check leaf content (non-black pixels)
        non_black = np.count_nonzero(np.sum(patch, axis=2))
        total = PATCH_SIZE * PATCH_SIZE
        leaf_ratio = non_black / total

        if leaf_ratio >= MIN_LEAF_RATIO:
            patches.append(patch)

    return patches

# =========================
# PROCESS DATASET
# =========================
def process_split(split):
    print(f"\nExtracting patches for {split} set...")

    in_root = os.path.join(SEG_DATASET, split)
    out_root = os.path.join(PATCH_DATASET, split)

    os.makedirs(out_root, exist_ok=True)

    for disease in os.listdir(in_root):
        in_dir = os.path.join(in_root, disease)
        out_dir = os.path.join(out_root, disease)
        os.makedirs(out_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(in_dir), desc=disease):
            img_path = os.path.join(in_dir, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            patches = extract_patches(image)

            base = os.path.splitext(img_name)[0]
            for i, patch in enumerate(patches):
                save_name = f"{base}_patch_{i}.jpg"
                save_path = os.path.join(out_dir, save_name)
                cv2.imwrite(save_path, patch)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    process_split("train")
    process_split("val")
    print("\nPatch extraction completed.")
