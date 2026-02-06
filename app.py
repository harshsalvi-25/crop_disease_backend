from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
import os
import gc

app = Flask(__name__)

# ============================
# LOAD TOMATO MODEL (ON START)
# ============================

MODEL_PATH = "tomato_model_final.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("tomato_model_final.keras not found in backend folder")

print("🔄 Loading Tomato model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Tomato model loaded")

# ============================
# CLASS LABELS (LOCKED ORDER)
# ============================

CLASS_NAMES = [
    "bacterial_spot",
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "mosaic_virus",
    "septoria_leaf_spot",
    "target_spot",
    "twospotted_spider_mite",
    "yellow_leaf_curl_virus"
]

# ============================
# SENSOR STORAGE
# ============================

latest_sensor = {
    "temperature": None,
    "humidity": None,
    "moisture": None
}

# ============================
# ROUTES
# ============================

@app.route("/")
def home():
    return "Tomato Backend Running"

# ---------- SENSOR DATA ----------
@app.route("/sensor", methods=["POST"])
def receive_sensor():
    latest_sensor["temperature"] = request.form.get("temperature")
    latest_sensor["humidity"] = request.form.get("humidity")
    latest_sensor["moisture"] = request.form.get("moisture")

    return jsonify({"status": "sensor data received"})

# ---------- IMAGE PREDICTION ----------
@app.route("/predict", methods=["POST"])
def predict():

    image_file = request.files.get("image")

    if not image_file:
        return jsonify({"error": "image missing"}), 400

    # SAME PIPELINE AS COLAB
    img = image.load_img(BytesIO(image_file.read()), target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx])
    label = CLASS_NAMES[idx]

    # HEALTHY SAFEGUARD
    if confidence < 0.60:
        label = "healthy"

    confidence = round(confidence * 100, 2)

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "sensor": latest_sensor
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
