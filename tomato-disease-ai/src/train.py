import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

# =============================
# BASIC SETTINGS
# =============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 15

TRAIN_DIR = r"C:\Users\PRATHAM\Ai-plant\tomato-disease-ai\dataset\train"
VAL_DIR   = r"C:\Users\PRATHAM\Ai-plant\tomato-disease-ai\dataset\val"

# =============================
# DATA GENERATORS (STRONGER AUGMENTATION)
# =============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.25,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print("Class indices:", train_gen.class_indices)

# =============================
# CLASS WEIGHTS (CRITICAL)
# =============================
class_counts = Counter(train_gen.classes)
total = sum(class_counts.values())

class_weights = {
    cls: total / (len(class_counts) * count)
    for cls, count in class_counts.items()
}

print("Class weights:", class_weights)

# =============================
# TWO-HEAD GENERATOR
# =============================
def two_head_generator(generator):
    while True:
        images, disease_labels = next(generator)

        # binary label: healthy=0, diseased=1
        binary_labels = (np.argmax(disease_labels, axis=1) != 0).astype(np.float32)

        yield images, {
            "binary_output": binary_labels,
            "disease_output": disease_labels
        }

# =============================
# FOCAL LOSS (FOR HARD DISEASE CASES)
# =============================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=1)
    return loss

# =============================
# MODEL DEFINITION
# =============================
base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)

# Head 1: Binary (Healthy vs Diseased)
binary_output = tf.keras.layers.Dense(
    1, activation="sigmoid", name="binary_output"
)(x)

# Head 2: Disease Classification
disease_output = tf.keras.layers.Dense(
    NUM_CLASSES, activation="softmax", name="disease_output"
)(x)

model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[binary_output, disease_output]
)

# =============================
# COMPILE — STAGE 1
# =============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss={
        "binary_output": "binary_crossentropy",
        "disease_output": categorical_focal_loss(gamma=2.0, alpha=0.25)
    },
    loss_weights={
        "binary_output": 0.3,
        "disease_output": 0.7
    },
    metrics={
        "binary_output": "accuracy",
        "disease_output": "accuracy"
    }
)

model.summary()

# =============================
# TRAIN — STAGE 1
# =============================
model.fit(
    two_head_generator(train_gen),
    validation_data=two_head_generator(val_gen),
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    epochs=EPOCHS_STAGE1
)

# =============================
# FINE-TUNING — DEEPER
# =============================
base_model.trainable = True

for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss={
        "binary_output": "binary_crossentropy",
        "disease_output": categorical_focal_loss(gamma=2.0, alpha=0.25)
    },
    loss_weights={
        "binary_output": 0.3,
        "disease_output": 0.7
    },
    metrics={
        "binary_output": "accuracy",
        "disease_output": "accuracy"
    }
)

model.fit(
    two_head_generator(train_gen),
    validation_data=two_head_generator(val_gen),
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    epochs=EPOCHS_STAGE2
)

# =============================
# SAVE MODEL
# =============================
model.save("two_head_tomato_disease_model_v2.keras")
print("Model saved successfully.")
