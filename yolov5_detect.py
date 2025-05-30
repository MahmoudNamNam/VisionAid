import torch
import torchvision.transforms as T
from PIL import Image
from io import BytesIO
from collections import Counter, defaultdict

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

COCO_CLASSES = model.names

# Predefined hazard classes from COCO
HAZARD_CLASSES = {
    "car", "bus", "truck", "motorcycle", "bicycle",
    "traffic light", "stop sign", "fire hydrant",
    "person", "train"
}


def detect_objects(image_bytes: bytes, categories=None):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        results = model(image)
        detections = results.pandas().xyxy[0]

        if categories:
            categories = [c.lower() for c in categories]
            detections = detections[detections['name'].str.lower().isin(categories)]

        output = []
        label_counter = Counter()
        spatial_map = defaultdict(list)
        hazards_detected = set()

        for _, row in detections.iterrows():
            label = row['name']
            label_counter[label] += 1

            x_center = (row['xmin'] + row['xmax']) / 2
            y_center = (row['ymin'] + row['ymax']) / 2

            if x_center < width / 3:
                h_pos = "on your left"
            elif x_center < 2 * width / 3:
                h_pos = "in front of you"
            else:
                h_pos = "on your right"

            if y_center < height / 3:
                v_pos = "above you"
            elif y_center < 2 * height / 3:
                v_pos = "at eye level"
            else:
                v_pos = "on the ground"

            box_area = (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])
            image_area = width * height
            area_ratio = box_area / image_area

            if area_ratio > 0.1:
                distance = "near"
            elif area_ratio < 0.02:
                distance = "far"
            else:
                distance = None

            position = f"{v_pos} and {h_pos}"
            if distance:
                position = f"{distance}, {position}"

            spatial_map[label].append(position)

            if label in HAZARD_CLASSES:
                hazards_detected.add(label)

            output.append({
                "label": label,
                "confidence": round(float(row['confidence']) * 100, 2),
                "box": [
                    float(row['xmin']),
                    float(row['ymin']),
                    float(row['xmax']),
                    float(row['ymax'])
                ]
            })

        description = ""
        if label_counter:
            description_parts = []
            for label, count in label_counter.items():
                positions = spatial_map[label]
                common_pos = Counter(positions).most_common(1)[0][0]
                noun = label + 's' if count > 1 else label
                description_parts.append(f"{count} {noun} {common_pos}")
            description = "You are seeing " + ', '.join(description_parts[:-1])
            if len(description_parts) > 1:
                description += f", and {description_parts[-1]}."
            else:
                description += f" {description_parts[0]}."
        else:
            description = "No recognizable objects detected."

        alerts = []
        for hazard in hazards_detected:
            alerts.append(f"Warning: {hazard} detected! Please be cautious.")

        return {
            "objects": output,
            "image_width": width,
            "image_height": height,
            "description": description,
            "alerts": alerts
        }

    except Exception as e:
        return {
            "objects": [],
            "image_width": 0,
            "image_height": 0,
            "description": f"Detection error: {str(e)}",
            "alerts": [f"Error: {str(e)}"]
        }
