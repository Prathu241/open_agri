import cv2
import numpy as np
import tensorflow as tf
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = "patch_disease_classifier.keras"
IMG_SIZE = 96
PATCHES_PER_IMAGE = 32
MIN_LEAF_RATIO = 0.6

# Ignore weak patch predictions
CONF_THRESHOLD = 0.6

CLASS_NAMES = [
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot"
]

DISEASE_CLASSES = [
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot"
]

# =========================
# LOAD MODEL
# =========================
print("Loading patch classifier...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.\n")

# =========================
# LEAF SEGMENTATION (HSV)
# =========================
def segment_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented

# =========================
# PATCH EXTRACTION
# =========================
def extract_patches(image):
    patches = []
    h, w, _ = image.shape

    if h < IMG_SIZE or w < IMG_SIZE:
        return patches

    for _ in range(PATCHES_PER_IMAGE * 3):
        if len(patches) >= PATCHES_PER_IMAGE:
            break

        x = np.random.randint(0, w - IMG_SIZE)
        y = np.random.randint(0, h - IMG_SIZE)

        patch = image[y:y + IMG_SIZE, x:x + IMG_SIZE]

        non_black = np.count_nonzero(np.sum(patch, axis=2))
        ratio = non_black / (IMG_SIZE * IMG_SIZE)

        if ratio >= MIN_LEAF_RATIO:
            patches.append(patch)

    return patches

# =========================
# SMART LEAF PREDICTION
# =========================
def predict_leaf(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("\n❌ ERROR: Image could not be loaded.")
        return "Invalid image", 0.0

    image = cv2.resize(image, (224, 224))
    segmented = segment_leaf(image)
    patches = extract_patches(segmented)

    if len(patches) == 0:
        return "No leaf detected", 0.0

    # Sum probabilities instead of voting
    prob_sums = {cls: 0.0 for cls in CLASS_NAMES}
    valid_patches = 0

    for patch in patches:
        patch = patch.astype("float32") / 255.0
        patch = np.expand_dims(patch, axis=0)

        probs = model.predict(patch, verbose=0)[0]
        max_prob = np.max(probs)

        # Ignore weak patches
        if max_prob < CONF_THRESHOLD:
            continue

        valid_patches += 1

        for i, cls in enumerate(CLASS_NAMES):
            prob_sums[cls] += probs[i]

    if valid_patches == 0:
        return "Uncertain", 0.0

    # Disease priority logic
    disease_scores = {d: prob_sums[d] for d in DISEASE_CLASSES}
    best_disease = max(disease_scores, key=disease_scores.get)

    if disease_scores[best_disease] > prob_sums["healthy"]:
        final_class = best_disease
        confidence = disease_scores[best_disease] / sum(prob_sums.values())
    else:
        final_class = "healthy"
        confidence = prob_sums["healthy"] / sum(prob_sums.values())

    return final_class, confidence

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("=== TOMATO DISEASE DETECTION ===\n")

    img_path = input("Enter image path: ").strip()
    img_path = img_path.strip('"').strip("'")

    disease, conf = predict_leaf(img_path)

    print("\n=== FINAL RESULT ===")
    print("Predicted Disease :", disease)
    print("Confidence        :", round(conf, 2))
