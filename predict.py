# predict.py
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_detection_model.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Preprocess the image
def preprocess_image(image_path, target_size=(96, 96)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Predict with heatmap
def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    confidence = np.max(prediction) * 100
    class_label = np.argmax(prediction, axis=1)[0]
    result = "Fake" if class_label == 0 else "Real"
    heatmap = np.mean(image[0], axis=2)  # Simple heatmap
    return result, confidence, heatmap

# Batch prediction
def batch_predict(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(directory, filename)
            result, confidence, _ = predict_image(image_path)
            results[filename] = (result, confidence)
    return results

# Example usage
if __name__ == "__main__":
    sample_dir = os.path.join(BASE_DIR, "real_and_fake_face_detection/real_and_fake_face/training_real")
    results = batch_predict(sample_dir)
    for filename, (result, confidence) in results.items():
        print(f"{filename}: {result} (Confidence: {confidence:.2f}%)")