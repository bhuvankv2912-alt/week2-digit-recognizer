import base64
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = tf.keras.models.load_model("mnist_model.h5")


def preprocess_image(base64_string: str) -> np.ndarray:
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    img = img.resize((28, 28), Image.Resampling.BILINEAR)

    img_array = np.array(img)
    # Invert colors if drawing is black stroke on white canvas
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    normalized = img_array / 255.0
    return normalized.reshape(1, 784)


@app.post("/predict")
async def predict(payload: dict):
    if "image" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'image' key.")

    processed_data = preprocess_image(payload["image"])
    predictions = model.predict(processed_data, verbose=0)[0]

    predicted_digit = int(np.argmax(predictions))
    confidence = float(predictions[predicted_digit])

    return {
        "prediction": predicted_digit,
        "confidence": round(confidence * 100, 2),
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return f.read()