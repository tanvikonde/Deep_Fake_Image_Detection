# train.py
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "real_and_fake_face_detection/real_and_fake_face")
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_detection_model.keras")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

# Data augmentation
data_with_aug = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=30,
    zoom_range=0.3,
    shear_range=0.3,
    brightness_range=[0.7, 1.3],
    rescale=1./255,
    validation_split=0.2
)

train = data_with_aug.flow_from_directory(
    DATASET_PATH,
    class_mode="binary",
    target_size=(96, 96),
    batch_size=32,
    subset="training",
    shuffle=True
)

val = data_with_aug.flow_from_directory(
    DATASET_PATH,
    class_mode="binary",
    target_size=(96, 96),
    batch_size=32,
    subset="validation",
    shuffle=False
)

# Compute class weights
class_counts = Counter(train.classes)
total_samples = sum(class_counts.values())
class_weights = {0: total_samples / (2 * class_counts[0]), 1: total_samples / (2 * class_counts[1])}
print(f"Class Weights: {class_weights}")

# Model
mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
tf.keras.backend.clear_session()

model = Sequential([
    mnet,
    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(2, activation="softmax")
])

# Unfreeze more layers
for layer in mnet.layers[-30:]:
    layer.trainable = True

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0005),  # Increased learning rate
    metrics=["accuracy"]
)
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001, verbose=1)
csv_logger = CSVLogger(os.path.join(BASE_DIR, 'training_log.csv'))

# Train
hist = model.fit(
    train,
    epochs=30,  # Increased epochs
    validation_data=val,
    callbacks=[early_stopping, reduce_lr, csv_logger],
    class_weight=class_weights,
    verbose=1
)

# Save model
model.save(MODEL_PATH)

# Plots
epochs = len(hist.history['loss'])
xc = range(epochs)

plt.figure(figsize=(10, 6))
plt.plot(xc, hist.history['loss'], label='Train Loss')
plt.plot(xc, hist.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'loss_plot.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(xc, hist.history['accuracy'], label='Train Accuracy')
plt.plot(xc, hist.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'accuracy_plot.png'))
plt.close()

# Confusion matrix
val_images, val_labels = next(val)
val_preds = model.predict(val_images)
val_preds_classes = np.argmax(val_preds, axis=1)

cm = confusion_matrix(val_labels, val_preds_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
plt.close()