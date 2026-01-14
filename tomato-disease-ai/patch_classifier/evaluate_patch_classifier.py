import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
MODEL_PATH = "patch_disease_classifier_finetuned.keras"
VAL_DIR = "../patch_dataset/val"
IMG_SIZE = 96
BATCH_SIZE = 32

CLASS_NAMES = [
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot"
]

# =========================
# LOAD MODEL
# =========================
print("Loading weighted patch classifier...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.\n")

# =========================
# LOAD VALIDATION DATA
# =========================
datagen = ImageDataGenerator(rescale=1.0 / 255)

val_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =========================
# RUN EVALUATION
# =========================
print("\nRunning evaluation on validation patches...\n")

pred_probs = model.predict(val_gen)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_gen.classes

print("=== CLASSIFICATION REPORT ===\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

print("\n=== CONFUSION MATRIX ===\n")
print(confusion_matrix(y_true, y_pred))
