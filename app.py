from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
import os

app = Flask(__name__)

# ============================
# LOAD TOMATO MODEL (NEW)
# ============================

MODEL_PATH = "tomato_model_v2.keras"   # <-- NEW MODEL NAME

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("tomato_model_v2.keras not found in backend folder")

print("🔄 Loading Tomato model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Tomato model loaded")

# ============================
# CLASS LABELS
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
# SENSOR RISK LOGIC
# ============================

def analyze_risk(temp, humidity, moisture):

    risks = []

    if humidity is None:
        return ["Sensor data not available"]

    if humidity > 85 and moisture > 70:
        risks.append("Favorable conditions for Late Blight")

    if humidity > 75 and temp > 20:
        risks.append("Possible Early Blight risk")

    if humidity > 80 and temp > 22:
        risks.append("Conditions favorable for Leaf Mold")

    if temp > 30 and humidity < 50:
        risks.append("Possible Spider Mites risk")

    if len(risks) == 0:
        risks.append("No major disease-favorable conditions detected")

    return risks

# ============================
# PRECAUTIONARY MEASURES
# ============================

PRECAUTIONS = {

    "late_blight": [
        "Remove infected leaves",
        "Avoid overhead irrigation",
        "Improve drainage",
        "Apply fungicide: Metalaxyl + Mancozeb",
        "Alternative: Carbendazim (Gretel)",
        "Use micronutrient spray (Microla) for plant strength"
    ],

    "early_blight": [
        "Remove infected plant debris",
        "Practice crop rotation",
        "Avoid wetting leaves",
        "Apply fungicide: Mancozeb or Chlorothalonil",
        "Alternative: Carbendazim + Mancozeb (Smooth)",
        "Apply foliar fertilizers to improve resistance"
    ],

    "leaf_mold": [
        "Reduce humidity",
        "Increase air circulation",
        "Avoid wet leaves",
        "Apply fungicide spray: Carbendazim",
        "Use bio-products: Bio Rapid / Verramicro",
        "Apply micronutrient spray (Microla)"
    ],

    "bacterial_spot": [
        "Use certified seeds",
        "Avoid overhead watering",
        "Apply bactericide: Kasugamycin + Copper Oxychloride (Conika)",
        "Use copper-based spray",
        "Apply micronutrients to boost immunity"
    ],

    "septoria_leaf_spot": [
        "Remove infected leaves",
        "Avoid overhead irrigation",
        "Apply fungicide: Mancozeb",
        "Alternative: Carbendazim based spray",
        "Use foliar fertilizer for plant recovery"
    ],

    "target_spot": [
        "Improve air circulation",
        "Remove infected leaves",
        "Apply fungicide: Chlorothalonil",
        "Use micronutrient fertilizer (Microla)",
        "Maintain balanced fertilization"
    ],

    "twospotted_spider_mite": [
        "Spray water on leaves",
        "Use neem oil",
        "Apply insecticide: Profex Super (Profenofos + Cypermethrin)",
        "Maintain proper irrigation",
        "Avoid excessive heat stress"
    ],

    "mosaic_virus": [
        "Remove infected plants immediately",
        "Control aphids (vector insects)",
        "Use resistant varieties",
        "Apply bio-stimulants (Bio Rapid)",
        "Maintain plant nutrition using micronutrients"
    ],

    "yellow_leaf_curl_virus": [
        "Control whiteflies",
        "Remove infected plants",
        "Use resistant varieties",
        "Apply insecticide for vector control",
        "Use micronutrient sprays for recovery"
    ],

    "healthy": [
        "Crop is healthy",
        "Maintain proper irrigation",
        "Apply balanced fertilizers",
        "Use foliar fertilizers periodically",
        "Regular monitoring recommended"
    ]
}

# ============================
# ROUTES
# ============================

@app.route("/")
def home():
    return "Tomato Backend Running (Fine Tuned Model)"

# ---------- SENSOR DATA ----------
@app.route("/sensor", methods=["POST"])
def receive_sensor():

    latest_sensor["temperature"] = float(request.form.get("temperature"))
    latest_sensor["humidity"] = float(request.form.get("humidity"))
    latest_sensor["moisture"] = float(request.form.get("moisture"))

    return jsonify({"status": "sensor data received"})

# ---------- IMAGE PREDICTION ----------
@app.route("/predict", methods=["POST"])
def predict():

    image_file = request.files.get("image")

    if not image_file:
        return jsonify({"error": "image missing"}), 400

    # EXACT SAME PIPELINE AS COLAB
    img = image.load_img(BytesIO(image_file.read()), target_size=(224,224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    index = int(np.argmax(pred))
    label = CLASS_NAMES[index]
    confidence = float(pred[0][index])

    confidence = round(confidence * 100, 2)

    risk = analyze_risk(
        latest_sensor["temperature"],
        latest_sensor["humidity"],
        latest_sensor["moisture"]
    )

    precautions = PRECAUTIONS.get(label, [])

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "sensor": latest_sensor,
        "risk": risk,
        "precautions": precautions
    })

# ============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
