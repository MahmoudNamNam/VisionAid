import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from typing import List, Tuple
import numpy as np
import json
import os

from schemas.schemas import DetectedObject
from utils.image_processing import get_torchvision_transform

# --- Configuration ---
MODEL_DIR = "models/weights" # ADJUST PATH if needed
# Ensure this directory exists and contains the model weights if downloaded manually
# Torchvision might download them automatically on first run if needed.

# --- Load Labels ---
COCO_LABELS_PATH = "data/coco_labels.json"
try:
    with open(COCO_LABELS_PATH) as f:
        COCO_LABELS = json.load(f)
except FileNotFoundError:
    print(f"ERROR: COCO labels file not found at {COCO_LABELS_PATH}")
    COCO_LABELS = [] # Fallback

# --- Global Variables ---
model = None
device = None
hazard_list = []
model_transform = None

def load_model():
    """Loads the Faster R-CNN model."""
    global model, device, model_transform
    if model is not None:
        return

    print("Loading Object Detection model (Faster R-CNN ResNet50 FPN)...")
    try:
        # Use recommended weights (currently V2)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT # Use .DEFAULT for latest
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

        # Set device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # Set model to evaluation mode

        # Get the preprocessing transforms associated with the weights
        model_transform = weights.transforms()
        print(f"Object Detection model loaded successfully on {device}.")

    except Exception as e:
        print(f"ERROR loading Object Detection model: {e}")
        model = None # Ensure model is None if loading failed


def set_hazard_list(hazards: List[str]):
    """Sets the list of hazards known to this module."""
    global hazard_list
    hazard_list = hazards
    print(f"Object Detection - Hazard list set: {hazard_list}")

def detect_objects(image: Image.Image, confidence_threshold: float = 0.5) -> List[DetectedObject]:
    """
    Detects objects using the loaded Faster R-CNN model.
    """
    if model is None or device is None or model_transform is None:
        print("ERROR: Object detection model not loaded.")
        return [] # Return empty list if model isn't ready

    if not COCO_LABELS:
        print("ERROR: COCO labels not loaded.")
        return []

    print(f"Detecting objects (Threshold: {confidence_threshold})...")
    results = []

    # Preprocess image using the model's required transforms
    input_tensor = model_transform(image).to(device)
    input_batch = [input_tensor] # Model expects a batch

    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model(input_batch)

    # Process outputs (outputs[0] contains results for the first image in the batch)
    # outputs[0] is a dict with keys 'boxes', 'labels', 'scores'
    pred_boxes = outputs[0]['boxes'].cpu().numpy()
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].cpu().numpy()

    # Filter results based on confidence threshold
    for box, label_idx, score in zip(pred_boxes, pred_labels, pred_scores):
        if score >= confidence_threshold:
            # Ensure label index is valid
            if 0 <= label_idx < len(COCO_LABELS):
                label_name = COCO_LABELS[label_idx]
            else:
                label_name = f"Unknown_Index_{label_idx}"
                print(f"Warning: Detected unknown label index: {label_idx}")

            # Convert box coordinates from float to int
            xmin, ymin, xmax, ymax = map(int, box)

            results.append(DetectedObject(
                label=label_name,
                confidence=float(score),
                box=[xmin, ymin, xmax, ymax]
            ))

    print(f"Detected {len(results)} objects meeting threshold.")
    return results

def identify_hazards(detected_objects: List[DetectedObject]) -> List[DetectedObject]:
    """Filters detected objects to identify hazards based on the hazard_list."""
    if not hazard_list:
        # print("Hazard list is empty, cannot identify hazards.")
        return [] # Don't identify if list isn't set

    hazards = [obj for obj in detected_objects if obj.label in hazard_list]
    # print(f"Identified {len(hazards)} hazards.")
    return hazards