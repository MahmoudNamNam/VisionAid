import torch
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

def detect_objects(image_bytes, selected_categories, conf_threshold=0.5):
    """
    Detects objects in an image based on selected categories and a confidence threshold.

    Args:
        image_bytes (bytes): The image data as bytes.
        selected_categories (list): A list of category names to detect.
        conf_threshold (float): The minimum confidence score for a detection to be included.

    Returns:
        tuple: A tuple containing:
            - list: A list of unique detected object labels (strings).
            - bytes: The annotated image as JPEG bytes.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)
        results = model(img_np)
        detections = results.pandas().xyxy[0]

        filtered = detections[
            (detections['name'].isin(selected_categories)) &
            (detections['confidence'] > conf_threshold)
        ]

        # Annotate image
        annotated_img_np = img_np.copy()  # Create a copy to avoid modifying the original
        for _, row in filtered.iterrows():
            x1, y1, x2, y2 = map(int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            label = row["name"]
            cv2.rectangle(annotated_img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Encode annotated image back to bytes
        _, encoded_img = cv2.imencode(".jpg", annotated_img_np)
        return filtered["name"].unique().tolist(), encoded_img.tobytes()

    except Exception as e:
        print(f"Error during object detection: {e}")
        return [], b""

if __name__ == '__main__':
    # Example usage (for testing this module directly)
    try:
        # Create a dummy image (replace with a real image path for testing)
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        pil_img = Image.fromarray(dummy_img)
        byte_io = BytesIO()
        pil_img.save(byte_io, 'JPEG')
        image_bytes = byte_io.getvalue()

        selected_categories = ["person", "car"]
        labels, annotated_image_bytes = detect_objects(image_bytes, selected_categories, conf_threshold=0.3)

        print("Detected labels:", labels)

        if annotated_image_bytes:
            with open("annotated_image.jpg", "wb") as f:
                f.write(annotated_image_bytes)
            print("Annotated image saved as annotated_image.jpg")
        else:
            print("No objects detected or error occurred during annotation.")

    except Exception as e:
        print(f"An error occurred in the example usage: {e}")