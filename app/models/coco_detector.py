import torch
from PIL import Image
import io
from typing import List, Dict, Optional

class CocoObjectDetector:
    """
    Wrapper for a COCO pre-trained object detection model (YOLOv5).
    """
    def __init__(self, model_name: str = 'yolov5s'):
        """
        Loads the pre-trained YOLOv5 model.
        Args:
            model_name (str): Name of the YOLOv5 model to load (e.g., 'yolov5s', 'yolov5m').
        """
        try:
            # Load the model from PyTorch Hub
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            self.model.eval() # Set model to evaluation mode
            print(f"Successfully loaded model '{model_name}'")
            # Get class names (YOLOv5 model has them internally)
            self.class_names = self.model.names
            print(f"Model classes: {len(self.class_names)}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.class_names = []

    def detect_objects(self, image_bytes: bytes, confidence_threshold: float = 0.25, selected_categories: Optional[List[str]] = None) -> List[Dict]:
        """
        Detects objects in a given image.

        Args:
            image_bytes (bytes): The input image as bytes.
            confidence_threshold (float): Minimum confidence score for detections.
            selected_categories (Optional[List[str]]): List of category names to filter by. If None, all categories are considered.

        Returns:
            List[Dict]: A list of detected objects, each with 'label', 'confidence', and 'box'. Returns empty list if model failed to load.
        """
        if not self.model:
            print("Model not loaded. Cannot perform detection.")
            return []

        try:
            # Load image from bytes
            img = Image.open(io.BytesIO(image_bytes))

            # Perform inference
            results = self.model(img)

            # Parse results using pandas DataFrame output
            predictions = results.pandas().xyxy[0] # Detections for image 0

            detected_objects = []

            # Convert selected category names to lowercase for case-insensitive matching
            allowed_categories = set(cat.lower() for cat in selected_categories) if selected_categories else None

            for index, row in predictions.iterrows():
                confidence = row['confidence']
                if confidence >= confidence_threshold:
                    label = row['name'].lower() # Use lowercase label for matching

                    # Apply category filtering
                    if allowed_categories is None or label in allowed_categories:
                        detected_objects.append({
                            "label": row['name'], # Keep original case for output
                            "confidence": float(f"{confidence:.2f}"), # Format confidence
                            "box": [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                        })

            return detected_objects

        except Exception as e:
            print(f"Error during object detection: {e}")
            return []
