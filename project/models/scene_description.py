import torch
import torchvision
from torchvision.models import ResNet50_Weights # Example using ResNet50
from PIL import Image
from typing import List, Tuple
import numpy as np
import os
import json

from utils.image_processing import get_torchvision_transform # Use appropriate transforms

# --- Configuration ---
MODEL_DIR = "models/weights" # ADJUST PATH
PLACES_MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_places365.pth") # ADJUST Filename
PLACES_LABELS_PATH = "data/places365_labels.txt"

# --- Global Variables ---
scene_model = None
device = None
places_labels = []
scene_transform = None

def load_scene_labels():
    """Loads Places365 labels from the text file."""
    global places_labels
    if not os.path.exists(PLACES_LABELS_PATH):
        print(f"ERROR: Places365 labels file not found: {PLACES_LABELS_PATH}")
        places_labels = []
        return

    try:
        with open(PLACES_LABELS_PATH, 'r') as f:
            # Assumes file has format like: /a/abbey 0 (label index) - adapt if needed
            # Or just the label name per line, corresponding to the index
            places_labels = [line.strip().split()[0] for line in f if line.strip()] # Example parsing
        print(f"Loaded {len(places_labels)} Places365 labels.")
    except Exception as e:
        print(f"ERROR loading Places365 labels: {e}")
        places_labels = []


def load_model():
    """
    Loads the Scene Recognition model (ResNet adapted for Places365).
    *** THIS IS A SIMPLIFIED/MOCK IMPLEMENTATION ***
    You MUST replace this with code that loads your specific Places365 model weights.
    """
    global scene_model, device, scene_transform
    if scene_model is not None:
        return

    load_scene_labels() # Load labels first
    if not places_labels:
        print("Cannot load scene model without labels.")
        return

    print("Loading Scene Description model (e.g., ResNet50 for Places365)...")
    # --- !!! IMPORTANT: Replace with your actual model loading !!! ---
    try:
        # Example: Assuming a ResNet50 adapted for Places365 (365 classes)
        # 1. Load a standard ResNet50 structure
        weights = ResNet50_Weights.DEFAULT # Use standard weights as base if needed
        scene_model = torchvision.models.resnet50(weights=None) # Load structure only

        # 2. Modify the final layer to match the number of Places365 classes
        num_ftrs = scene_model.fc.in_features
        scene_model.fc = torch.nn.Linear(num_ftrs, len(places_labels)) # Adjust output layer

        # 3. Load your downloaded Places365 weights
        if not os.path.exists(PLACES_MODEL_PATH):
             print(f"ERROR: Places365 model weights not found at {PLACES_MODEL_PATH}")
             print("--- MOCKING Scene Model ---")
             # Keep scene_model as None or a mock flag to indicate failure/mocking
             scene_model = "MOCK_MODEL" # Use a string flag for mocked state
             return # Cannot proceed without weights

        # Load the state dict (weights) - ensure it matches the model structure
        # checkpoint = torch.load(PLACES_MODEL_PATH, map_location=torch.device('cpu')) # Load weights
        # scene_model.load_state_dict(checkpoint['state_dict']) # Adapt based on your weight file format

        # -------------------------------------------------------------

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # scene_model.to(device) # Uncomment when using real model
        # scene_model.eval()     # Uncomment when using real model

        # Define the transforms needed for this model (likely similar to ImageNet)
        # scene_transform = weights.transforms() # Use transforms from base weights if appropriate
        # OR define custom transforms needed by Places365 model
        scene_transform = get_torchvision_transform() # Placeholder

        print(f"Scene Description model loaded successfully structure on {device}.")
        print(f"!!! WARNING: Actual weights from {PLACES_MODEL_PATH} need to be loaded and verified.")


    except Exception as e:
        print(f"ERROR loading Scene Description model: {e}")
        scene_model = None


def describe_scene(image: Image.Image) -> Tuple[str, float]:
    """
    Classifies the scene using the loaded model.
    *** RETURNS MOCK DATA IF MODEL ISN'T LOADED ***
    """
    # --- MOCK IMPLEMENTATION ---
    if scene_model == "MOCK_MODEL" or scene_model is None or not places_labels:
        print("--- Returning MOCK Scene Description ---")
        # Find a plausible default label if available
        default_label = "park" if "park" in places_labels else places_labels[0] if places_labels else "unknown_scene"
        return default_label, 0.50 # Mock label and confidence
    # --- End MOCK ---

    # --- Real Implementation (when model is loaded) ---
    # if scene_model is None or device is None or scene_transform is None or not places_labels:
    #     print("ERROR: Scene description model/labels not ready.")
    #     return "unknown_scene", 0.0

    print("Describing scene...")
    results = []

    # Preprocess image
    input_tensor = scene_transform(image).to(device)
    input_batch = [input_tensor] # Create a mini-batch as expected by the model

    with torch.no_grad():
        outputs = scene_model(input_batch) # Get model outputs (logits)
        # Apply Softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # Get the top prediction
        top_prob, top_catid = torch.topk(probabilities, 1)
        pred_index = top_catid[0].item()
        confidence = top_prob[0].item()

    if 0 <= pred_index < len(places_labels):
        scene_label = places_labels[pred_index]
    else:
        scene_label = "unknown_scene"
        print(f"Warning: Predicted scene index out of bounds: {pred_index}")
        confidence = 0.0 # Low confidence for unknown index

    print(f"Scene detected: '{scene_label}' (Confidence: {confidence:.2f})")
    return scene_label, confidence