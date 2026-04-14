from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------
# INIT APP
# -------------------------------
app = Flask(__name__)

# -------------------------------
# LOAD MODEL (runs once)
# -------------------------------
try:
    model = load_model("model.h5")
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)
    model = None

# -------------------------------
# CLASS NAMES
# -------------------------------
class_names = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy"
]

# -------------------------------
# GLOBAL RESULT
# -------------------------------
latest_result = {
    "disease": "No data yet",
    "confidence": 0,
    "cure": "Waiting for image"
}

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_disease(img_path):
    if model is None:
        return "Model_Error", 0.0

    img = cv2.imread(img_path)

    if img is None:
        return "Image_Error", 0.0

    try:
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.reshape(img, (1, 224, 224, 3))

        pred = model.predict(img, verbose=0)

        class_index = np.argmax(pred)
        confidence = float(np.max(pred)) * 100

        # avoid unrealistic 100%
        confidence = min(confidence, 99.9)

        disease = class_names[class_index]

        return disease, confidence

    except Exception as e:
        print("Prediction error:", e)
        return "Prediction_Error", 0.0

# -------------------------------
# CURE FUNCTION
# -------------------------------
def get_cure(disease):
    cures = {
        "Bacterial_spot": "Use copper-based bactericide, avoid overhead watering",
        "Early_blight": "Apply fungicide and remove infected leaves",
        "Late_blight": "Use chlorothalonil fungicide immediately",
        "Leaf_Mold": "Improve ventilation and apply fungicide",
        "Septoria_leaf_spot": "Remove affected leaves and spray fungicide",
        "Spider_mites": "Use neem oil or insecticidal soap",
        "Target_Spot": "Apply appropriate fungicide",
        "Yellow_Leaf_Curl_Virus": "Control whiteflies, remove infected plants",
        "Tomato_mosaic_virus": "Remove infected plants, sanitize tools",
        "healthy": "No action needed"
    }

    return cures.get(disease, "Consult agricultural expert")

# -------------------------------
# UPLOAD API (ESP32)
# -------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file_data = request.data

        if not file_data:
            return jsonify({"error": "No data received"}), 400

        file_path = "image.jpg"

        with open(file_path, "wb") as f:
            f.write(file_data)

        print("Image received!")

        disease, confidence = predict_disease(file_path)

        global latest_result
        latest_result = {
            "disease": disease,
            "confidence": round(confidence, 2),
            "cure": get_cure(disease)
        }

        print("Prediction:", latest_result)

        return jsonify({"status": "ok"})

    except Exception as e:
        print("Upload error:", e)
        return jsonify({"error": "Upload failed"}), 500

# -------------------------------
# RESULT API (MOBILE)
# -------------------------------
@app.route('/latest', methods=['GET'])
def latest():
    return jsonify(latest_result)

# -------------------------------
# HEALTH CHECK (optional)
# -------------------------------
@app.route('/', methods=['GET'])
def home():
    return "ESP32 AI Server Running"

# -------------------------------
# RUN SERVER (RENDER COMPATIBLE)
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)