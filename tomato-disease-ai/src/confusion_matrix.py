import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================
# PATHS
# =============================
MODEL_PATH = "two_head_tomato_disease_model_v2.keras"
VAL_DIR = r"C:\Users\PRATHAM\Ai-plant\tomato-disease-ai\dataset\val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =============================
# LOAD MODEL
# =============================
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# =============================
# LOAD VALIDATION DATA
# =============================
datagen = ImageDataGenerator(rescale=1./255)

val_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(val_gen.class_indices.keys())
print("Classes:", class_names)

# =============================
# PREDICTIONS
# =============================
binary_preds, disease_preds = model.predict(val_gen, verbose=1)

y_true = val_gen.classes
y_pred = np.argmax(disease_preds, axis=1)

# =============================
# CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(8, 8))
disp.plot(cmap="Blues", values_format="d")
plt.title("Disease Classification Confusion Matrix")

# 🔴 SAVE INSTEAD OF SHOW
plt.savefig("confusion_matrix.png")
plt.close()

print("Confusion matrix saved as confusion_matrix.png")
