import os
from predict_leaf import predict_leaf

# =========================
# CONFIG
# =========================
VAL_DIR = "../segmented_dataset_v2/val"

CLASSES = [
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot"
]

# =========================
# EVALUATION
# =========================
total = 0
correct = 0
class_correct = {c: 0 for c in CLASSES}
class_total = {c: 0 for c in CLASSES}

print("\n=== LEAF LEVEL EVALUATION STARTED ===\n")

for cls in CLASSES:
    cls_dir = os.path.join(VAL_DIR, cls)
    if not os.path.exists(cls_dir):
        continue

    for img_name in os.listdir(cls_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(cls_dir, img_name)

        pred, conf = predict_leaf(img_path)

        total += 1
        class_total[cls] += 1

        if pred == cls:
            correct += 1
            class_correct[cls] += 1

        print(f"{img_name} | GT: {cls} | Pred: {pred} | Conf: {conf:.2f}")

# =========================
# RESULTS
# =========================
print("\n=== FINAL LEAF LEVEL RESULTS ===")

accuracy = correct / total if total > 0 else 0
print(f"\nOverall Leaf Accuracy: {accuracy * 100:.2f}% ({correct}/{total})\n")

for cls in CLASSES:
    if class_total[cls] > 0:
        acc = class_correct[cls] / class_total[cls]
        print(f"{cls:20s}: {acc * 100:.2f}% ({class_correct[cls]}/{class_total[cls]})")

print("\n=== EVALUATION COMPLETE ===")
