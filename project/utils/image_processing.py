from PIL import Image
import io
import numpy as np
import torch # For tensor conversion if needed by models
from torchvision import transforms as T # For torchvision transforms

# --- Standard Torchvision Transforms ---
# Usually required for models trained with torchvision
def get_torchvision_transform():
    return T.Compose([
        T.ToTensor() # Converts PIL image (H x W x C) to tensor (C x H x W) in [0.0, 1.0]
        # Add normalization if the model requires it (check model docs)
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Image Loading ---
async def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Loads image from bytes using PIL."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if it's not (e.g., RGBA, P, L)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        raise ValueError("Could not process the uploaded image file.")

# --- Preprocessing (Example for Torchvision) ---
def preprocess_for_torchvision(image: Image.Image):
    """Applies standard torchvision transforms."""
    transform = get_torchvision_transform()
    return transform(image)
