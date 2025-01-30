import base64
import os
import gdown
from io import BytesIO
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image

# Define constants
MODEL_FILENAME = "model.h5"  # Model file name
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)  # Local model path
GDRIVE_MODEL_ID = "1-jKXwRy9tFvERfTBWLLsni_vKN63srt_"  # Replace with your actual Google Drive file ID
IMG_SIZE = 224  # Model input size

# Function to download model from Google Drive if not found
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}", MODEL_PATH, quiet=False)
    else:
        print("Model already exists locally.")

# Ensure model is available
download_model()

# Load the model
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class ImageRequest(BaseModel):
    base64_image: str  # Accepts Base64-encoded image

# Function to convert Base64 string to an image
def base64_to_image(base64_string):
    """
    Converts a Base64 string to an image and saves it locally.
    """
    image_data = base64.b64decode(base64_string)  # Decode Base64 string
    image = Image.open(BytesIO(image_data))  # Convert byte data to an image
    image = image.convert("RGB")  # Ensure the image is in RGB mode
    image.save("temp_image.png", "PNG")  # Save as PNG
    return "temp_image.png"  # Return saved image path

# Function to process image for model prediction
def process_image(image_path):
    """
    Preprocess the image for TensorFlow model prediction.
    """
    image = tf.io.read_file(image_path)  # Read image
    image = tf.image.decode_png(image, channels=3)  # Decode PNG
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])  # Resize
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction endpoint
@app.post("/predict")
async def predict(image_request: ImageRequest):
    """
    API endpoint to receive Base64 image and return predictions.
    """
    image_path = base64_to_image(image_request.base64_image)  # Convert Base64 → Image
    processed_image = process_image(image_path)  # Preprocess image

    predictions = MODEL.predict(processed_image)  # Get predictions
    max_confidence = float(np.max(predictions))  # Highest confidence
    predicted_class_index = int(np.argmax(predictions))  # Predicted class index

    return {"class_index": predicted_class_index, "confidence": max_confidence}
