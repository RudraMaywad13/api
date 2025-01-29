import base64
from io import BytesIO
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image
import os

model_path = os.path.join(os.getcwd(), "20250126-05181737868724-full-image-set-mobilenetv2-Adam.h5")

if os.path.exists(model_path):
    MODEL = tf.keras.models.load_model(model_path)
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Define image size (should match the model's input size)
IMG_SIZE = 224

# Initialize FastAPI app
app = FastAPI()

class ImageRequest(BaseModel):
    base64_image: str  # Accepts Base64-encoded image

def base64_to_image(base64_string):
    """
    Converts a Base64 string to an image and returns the file path.
    """
    image_data = base64.b64decode(base64_string)  # Decode Base64
    image = Image.open(BytesIO(image_data))  # Convert to Image
    image = image.convert("RGB")  # Ensure it's RGB
    image.save("temp_image.png", "PNG")  # Save as PNG
    return "temp_image.png"  # Return saved file path

def process_image(image_path):
    """
    Preprocess image for TensorFlow model (resize, normalize).
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Decode PNG
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize (0-1)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])  # Resize
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict")
async def predict(image_request: ImageRequest):
    """
    API endpoint to receive Base64 image and return predictions.
    """
    image_path = base64_to_image(image_request.base64_image)  # Convert Base64 → PNG
    processed_image = process_image(image_path)  # Preprocess image

    predictions = MODEL.predict(processed_image)  # Get predictions
    max_confidence = float(np.max(predictions))  # Get highest confidence
    predicted_class_index = int(np.argmax(predictions))  # Get class index

    return {"class_index": predicted_class_index, "confidence": max_confidence}
