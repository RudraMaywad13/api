import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Define constants
MODEL_PATH = "20250126-05181737868724-full-image-set-mobilenetv2-Adam.h5"
IMG_SIZE = 224

# Check if the model exists before loading
if os.path.exists(MODEL_PATH):
    MODEL = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Initialize FastAPI app
app = FastAPI()

# Request model
class ImageRequest(BaseModel):
    base64_image: str  # Accepts Base64-encoded image

def base64_to_image(base64_string):
    """Converts a Base64 string to a PIL image (processed in memory)."""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image.convert("RGB")  # Ensure it's RGB
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def process_image(image):
    """Preprocess image for TensorFlow model (resize, normalize)."""
    image = image.resize((IMG_SIZE, IMG_SIZE))  # Resize
    image_array = np.array(image) / 255.0  # Normalize (0-1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.post("/predict")
async def predict(image_request: ImageRequest):
    """API endpoint to receive Base64 image and return predictions."""
    image = base64_to_image(image_request.base64_image)  # Decode Base64 → Image
    processed_image = process_image(image)  # Preprocess image

    # Get predictions
    predictions = MODEL.predict(processed_image)
    predicted_class_index = int(np.argmax(predictions))  # Get class index
    max_confidence = float(np.max(predictions))  # Get confidence score

    return {"class_index": predicted_class_index, "confidence": max_confidence}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Auto-detect PORT for Render
    uvicorn.run(app, host="0.0.0.0", port=port)
