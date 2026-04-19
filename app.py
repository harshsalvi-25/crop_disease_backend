from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
import os

app = Flask(__name__)

# ============================
# LOAD MODEL
# ============================

MODEL_PATH = "tomato_model_v2.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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
# RISK ANALYSIS
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

    if not risks:
        risks.append("No major disease-favorable conditions detected")

    return risks

# ============================
# PRECAUTIONS (CLEAN)
# ============================

PRECAUTIONS = {
    "late_blight": [
        "Remove infected leaves",
        "Avoid overhead irrigation",
        "Improve drainage",
        "Maintain proper plant spacing",
        "Monitor crop regularly"
    ],
    "early_blight": [
        "Remove infected plant debris",
        "Practice crop rotation",
        "Avoid wetting leaves",
        "Ensure good air circulation",
        "Monitor crop regularly"
    ],
    "leaf_mold": [
        "Reduce humidity",
        "Increase air circulation",
        "Avoid wet leaves",
        "Maintain proper spacing between plants",
        "Monitor crop regularly"
    ],
    "bacterial_spot": [
        "Use certified seeds",
        "Avoid overhead watering",
        "Remove infected leaves",
        "Maintain field hygiene",
        "Monitor crop regularly"
    ],
    "septoria_leaf_spot": [
        "Remove infected leaves",
        "Avoid overhead irrigation",
        "Maintain plant spacing",
        "Keep field clean",
        "Monitor crop regularly"
    ],
    "target_spot": [
        "Improve air circulation",
        "Remove infected leaves",
        "Maintain proper spacing",
        "Keep field clean",
        "Monitor crop regularly"
    ],
    "twospotted_spider_mite": [
        "Spray water on leaves",
        "Maintain proper irrigation",
        "Reduce plant stress",
        "Avoid excessive heat conditions",
        "Monitor crop regularly"
    ],
    "mosaic_virus": [
        "Remove infected plants immediately",
        "Control insect vectors",
        "Use resistant varieties",
        "Maintain field hygiene",
        "Monitor crop regularly"
    ],
    "yellow_leaf_curl_virus": [
        "Control whiteflies",
        "Remove infected plants",
        "Use resistant varieties",
        "Maintain field hygiene",
        "Monitor crop regularly"
    ],
    "healthy": [
        "Crop is healthy",
        "Maintain proper irrigation",
        "Ensure balanced nutrition",
        "Regular monitoring recommended",
        "Maintain field hygiene"
    ]
}

# ============================
# TREATMENT (WITH IMAGES)
# ============================

TREATMENT = {
    "bacterial_spot": {
        "pesticide": [
            {"name": "Conika Fungicide (Kasugamycin 5% + Copper Oxychloride 45%)",
             "image": "https://dujjhct8zer0r.cloudfront.net/media/prod_image/16061044081762166595.webp"}
        ],
        "fertilizer": [
            {"name": "Microla Micronutrient Fertilizer",
             "image": "https://dujjhct8zer0r.cloudfront.net/media/prod_image/d49128f276a7167a74084e625b95d3ce-09-08-23-10-08-34.webp"}
        ]
    },

    "early_blight": {
        "pesticide": [
            {"name": "Smooth Fungicide (Carbendazim 12% + Mancozeb 63% WP)",
             "image": "https://agribegri.com/_next/image?url=https://dujjhct8zer0r.cloudfront.net/media/prod_image/927b3b4fa489e1ee44db16697765d11d.webp"}
        ],
        "fertilizer": [
            {"name": "Verramicro Micronutrient Fertilizer",
             "image": "https://verrafert.com/wp-content/uploads/2024/06/VERRAMICRO-scaled-Photoroom.jpg"}
        ]
    },

    "late_blight": {
        "pesticide": [
            {"name": "Smooth Fungicide (Carbendazim 12% + Mancozeb 63% WP)",
             "image": "https://agribegri.com/_next/image?url=https://dujjhct8zer0r.cloudfront.net/media/prod_image/927b3b4fa489e1ee44db16697765d11d.webp"}
        ],
        "fertilizer": [
            {"name": "Bio Rapid Bio-Fertilizer",
             "image": "https://5.imimg.com/data5/SELLER/Default/2024/6/425922715/OM/NM/SU/199757407/250ml-harit-sanjivani-new-rapid-bio-fertilizer.jpeg"}
        ]
    },

    "leaf_mold": {
        "pesticide": [
            {"name": "Gretel Fungicide (Carbendazim 50% WP)",
             "image": "https://www.nichinoindia.com/assets/img/gretel/gretel.png"}
        ],
        "fertilizer": [
            {"name": "Microla Micronutrient Fertilizer",
             "image": "https://dujjhct8zer0r.cloudfront.net/media/prod_image/d49128f276a7167a74084e625b95d3ce-09-08-23-10-08-34.webp"}
        ]
    },

    "septoria_leaf_spot": {
        "pesticide": [
            {"name": "Mancozeb Fungicide (Mancozeb 75% WP)",
             "image": "https://5.imimg.com/data5/SELLER/Default/2021/7/QW/GH/SA/6616513/mancozeb-75-wp-contact-fungicide.jpg"}
        ],
        "fertilizer": [
            {"name": "Foliar Fertilizer Spray",
             "image": "https://m.media-amazon.com/images/I/71L28-kC0jL._AC_UF1000,1000_QL80_.jpg"}
        ]
    },

    "target_spot": {
        "pesticide": [
            {"name": "Chlorothalonil Fungicide (Chlorothalonil 75% WP)",
             "image": "https://dujjhct8zer0r.cloudfront.net/media/prod_image/17500355111758080743.webp"}
        ],
        "fertilizer": [
            {"name": "Verramicro Micronutrient Fertilizer",
             "image": "https://verrafert.com/wp-content/uploads/2024/06/VERRAMICRO-scaled-Photoroom.jpg"}
        ]
    },

    "twospotted_spider_mite": {
        "pesticide": [
            {"name": "Profex Super Insecticide (Profenofos 40% + Cypermethrin 4%)",
             "image": "https://cultree.in/cdn/shop/files/ProfexSuper.jpg"}
        ],
        "fertilizer": [
            {"name": "Foliar Fertilizer Spray",
             "image": "https://m.media-amazon.com/images/I/71L28-kC0jL._AC_UF1000,1000_QL80_.jpg"}
        ]
    },

    "mosaic_virus": {
        "pesticide": [
            {"name": "Imidacloprid Insecticide (Imidacloprid 17.8% SL)",
             "image": "https://dujjhct8zer0r.cloudfront.net/media/prod_image/9177795341754294797.webp"}
        ],
        "fertilizer": [
            {"name": "Bio Rapid Bio-Fertilizer",
             "image": "https://5.imimg.com/data5/SELLER/Default/2024/6/425922715/OM/NM/SU/199757407/250ml-harit-sanjivani-new-rapid-bio-fertilizer.jpeg"}
        ]
    },

    "yellow_leaf_curl_virus": {
        "pesticide": [
            {"name": "Imidacloprid Insecticide (Imidacloprid 17.8% SL)",
             "image": "https://5.imimg.com/data5/SELLER/Default/2021/4/ZO/YV/KA/12143645/imidacloprid-insecticide-500x500.jpg"}
        ],
        "fertilizer": [
            {"name": "Microla Micronutrient Fertilizer",
             "image": "https://dujjhct8zer0r.cloudfront.net/media/prod_image/d49128f276a7167a74084e625b95d3ce-09-08-23-10-08-34.webp"}
        ]
    },

    "healthy": {
        "pesticide": [
            {"name": "Not Required",
             "image": "https://plantperfect.com/wp-content/uploads/2022/07/Plant-Perfect-Garden-Center-How-to-Care-for-Your-Tomatoes-in-Bismarck-healthy-tomatoes-on-vine.jpg"}
        ],
        "fertilizer": [
            {"name": "Microla Micronutrient Fertilizer",
             "image": "https://dujjhct8zer0r.cloudfront.net/media/prod_image/d49128f276a7167a74084e625b95d3ce-09-08-23-10-08-34.webp"}
        ]
    }
}

# ============================
# ROUTES
# ============================

@app.route("/")
def home():
    return "Backend Running"

@app.route("/sensor", methods=["POST"])
def receive_sensor():
    latest_sensor["temperature"] = float(request.form.get("temperature"))
    latest_sensor["humidity"] = float(request.form.get("humidity"))
    latest_sensor["moisture"] = float(request.form.get("moisture"))
    return jsonify({"status": "sensor data received"})

@app.route("/predict", methods=["POST"])
def predict():

    image_file = request.files.get("image")

    if not image_file:
        return jsonify({"error": "image missing"}), 400

    img = image.load_img(BytesIO(image_file.read()), target_size=(224,224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    index = int(np.argmax(pred))
    label = CLASS_NAMES[index]
    confidence = round(float(pred[0][index]) * 100, 2)

    risk = analyze_risk(
        latest_sensor["temperature"],
        latest_sensor["humidity"],
        latest_sensor["moisture"]
    )

    precautions = PRECAUTIONS.get(label, [])
    treatment = TREATMENT.get(label, {"pesticide": [], "fertilizer": []})

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "sensor": latest_sensor,
        "risk": risk,
        "precautions": precautions,
        "treatment": treatment
    })

# ============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
