import torch
import cv2
import numpy as np
import os
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RAW_DATASET = "../dataset"
OUT_DATASET = "../segmented_dataset_v2"

IMG_SIZE = 512  # input image size

# =========================
# LOAD MODEL
# =========================
print("Loading SegFormer model...")

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

model.to(DEVICE)
model.eval()

print("SegFormer loaded on", DEVICE)

# =========================
# SEGMENTATION FUNCTION
# =========================
def segment_leaf(image_bgr):
    # Convert to RGB and PIL
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    # Preprocess
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Get segmentation map (low-res)
    logits = outputs.logits  # [1, num_classes, h, w]
    seg_map = torch.argmax(logits, dim=1)[0].cpu().numpy()

    # Resize mask to image size
    leaf_mask = cv2.resize(
        seg_map.astype(np.uint8),
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Treat non-background as leaf
    leaf_mask = (leaf_mask != 0).astype(np.uint8) * 255

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    # Apply mask
    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=leaf_mask)

    return segmented

# =========================
# PROCESS DATASET
# =========================
def process_split(split):
    in_root = os.path.join(RAW_DATASET, split)
    out_root = os.path.join(OUT_DATASET, split)

    os.makedirs(out_root, exist_ok=True)

    for disease in os.listdir(in_root):
        in_dir = os.path.join(in_root, disease)
        out_dir = os.path.join(out_root, disease)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing {split}/{disease} ...")

        for img_name in os.listdir(in_dir):
            img_path = os.path.join(in_dir, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            segmented = segment_leaf(image)

            cv2.imwrite(os.path.join(out_dir, img_name), segmented)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("Starting SegFormer segmentation...")
    process_split("train")
    process_split("val")
    print("Segmentation completed successfully.")
