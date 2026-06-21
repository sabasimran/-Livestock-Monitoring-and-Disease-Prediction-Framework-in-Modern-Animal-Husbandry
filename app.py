import os
import time
import serial
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename


app = Flask(__name__)



SERIAL_PORT = "COM3"
BAUD_RATE   = 9600



UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



MODEL_PATH = "model.h5"
model      = load_model(MODEL_PATH)
print("Model loaded.")



CLASSES = {
    0: "Healthy",
    1: "Lumpy"
}



DISEASE_INFO = {
    "Healthy": {
        "label":       "Healthy Animal",
        "severity":    "None",
        "color":       "#16a34a",
        "description": "The livestock appears healthy with no visible signs of disease or skin abnormalities.",
        "action":      "No intervention required. Continue regular feeding, vaccination schedule, and routine health monitoring.",
        "icon":        "🟢"
    },
    "Lumpy": {
        "label":       "Lumpy Skin Disease",
        "severity":    "High",
        "color":       "#dc2626",
        "description": "Lumpy Skin Disease (LSD) detected. Caused by the Lumpy Skin Disease Virus (LSDV), this is characterised by firm, raised nodules (2–5 cm) on the skin, fever, and reduced milk production. It spreads rapidly through insect vectors.",
        "action":      "Immediately isolate the affected animal. Notify a veterinarian for confirmation and treatment. Administer supportive care (anti-inflammatory, antibiotics for secondary infections). Implement vector control measures and vaccinate the remaining herd.",
        "icon":        "🔴"
    }
}



SERIAL_MESSAGES = {
    "Healthy": "D2",
    "Lumpy":   "D1"
}



last_prediction = {}



def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def predict_image(img_path):
    img      = image.load_img(img_path, target_size=(224, 224))
    arr      = image.img_to_array(img) / 255.0
    arr      = np.expand_dims(arr, axis=0)
    preds    = model.predict(arr)
    idx      = int(np.argmax(preds[0]))
    conf     = float(np.max(preds[0])) * 100
    cls      = CLASSES[idx]
    all_conf = {CLASSES[i]: float(preds[0][i]) * 100 for i in range(len(CLASSES))}
    return cls, conf, all_conf


def send_serial_message(prediction):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(1.5)                         # wait for Arduino to initialize
        message = SERIAL_MESSAGES.get(prediction, "Unknown")
        ser.write((message).encode())
        ser.close()
        print(f"Serial sent: {message} for {prediction}")
    except serial.SerialException as e:
        print(f"Serial error: {e}")


# ── Routes ──
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file selected.")
        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", error="No file selected.")
        if not allowed_file(file.filename):
            return render_template("upload.html", error="Invalid file type. Use PNG, JPG, or JPEG.")

        filename  = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename  = timestamp + filename
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        prediction, confidence, all_conf = predict_image(save_path)

        # Send D1 (Healthy) / D2 (Lumpy) signal to IoT device
        send_serial_message(prediction)

        info = DISEASE_INFO[prediction]

        global last_prediction
        last_prediction = {
            "prediction":  prediction,
            "label":       info["label"],
            "confidence":  round(confidence, 2),
            "severity":    info["severity"],
            "color":       info["color"],
            "description": info["description"],
            "action":      info["action"],
            "icon":        info["icon"],
            "serial":      SERIAL_MESSAGES.get(prediction),
            "image_url":   f"/static/uploads/{filename}",
            "all_conf":    {CLASSES[i]: round(all_conf[CLASSES[i]], 2) for i in range(len(CLASSES))},
            "timestamp":   datetime.now().strftime("%d %b %Y, %I:%M %p")
        }

        return redirect(url_for("result"))
    return render_template("upload.html")


@app.route("/result")
def result():
    if not last_prediction:
        return redirect(url_for("upload"))
    return render_template("result.html", data=last_prediction)


if __name__ == "__main__":
    print("Livestock Health Monitoring Server...")
    print("Server running on http://127.0.0.1:5004")
    app.run(host="0.0.0.0", port=5004, debug=False)