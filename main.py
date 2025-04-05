from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
from detector import detect_objects
from my_utils import generate_speech
import uvicorn
import base64

app = FastAPI()

# Category groups
category_groups = {
    "People": ["person"],
    "Vehicles": ["car", "bus", "truck", "motorcycle", "bicycle"],
    "Furniture": ["chair", "couch", "bed", "dining table"]
}

@app.post("/detect/")
async def detect(
    file: UploadFile = File(...),
    lang: Optional[str] = Form("en"),
    selected_groups: Optional[str] = Form("People,Vehicles,Furniture"),
    conf: Optional[float] = Form(0.5)
):
    image_bytes = await file.read()

    # If selected_groups is None or empty, fall back to default
    group_names = selected_groups.split(",") if selected_groups else ["People", "Vehicles", "Furniture"]
    selected_categories = [item for group in group_names for item in category_groups.get(group.strip(), [])]

    labels, annotated_img_bytes = detect_objects(image_bytes, selected_categories, conf_threshold=conf or 0.5)
    audio_base64 = generate_speech(labels, lang or "en")

    # Base64 encode the annotated image bytes
    annotated_image_base64 = base64.b64encode(annotated_img_bytes).decode('utf-8')

    return JSONResponse({
        "labels": labels,
        "audio": audio_base64,
        "annotated_image": annotated_image_base64  # Added annotated image
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)