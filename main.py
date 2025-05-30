from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from scene_model import classify_scene
import uvicorn
import logging
from typing import List
from yolov5_detect import detect_objects
import cv2
import pytesseract
import numpy as np
import json
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Scene Classifier API for Blind Assistance")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

logging.basicConfig(filename="unknown_labels.log", level=logging.INFO)

SCENE_JSON_PATH = os.path.join(os.path.dirname(__file__), "scene_descriptions.json")

with open(SCENE_JSON_PATH, "r", encoding="utf-8") as f:
    scene_descriptions = json.load(f)


@app.post("/detect-objects")
async def detect_objects_api(file: UploadFile = File(...), categories: List[str] = Form(None)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File is not an image.")

        image_bytes = await file.read()
        detection_result = detect_objects(image_bytes, categories)

        return JSONResponse(content=detection_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scene-detection")
async def classify_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File is not an image.")

        image_bytes = await file.read()
        predictions = classify_scene(image_bytes, top_n=3)

        if not predictions:
            raise HTTPException(status_code=500, detail="Could not classify the image.")

        fallback = False
        top_label, top_prob = predictions[0]
        readable_label = top_label.replace('_', ' ')

        if top_prob < 0.4:
            fallback = True

        if not fallback and readable_label in scene_descriptions:
            desc = scene_descriptions[readable_label]
        else:
            fallback_labels = [label.replace('_', ' ') for label, _ in predictions[:3]]
            desc = f"This image may show {', '.join(fallback_labels[:-1])}, or {fallback_labels[-1]}."
            for label in fallback_labels:
                if label not in scene_descriptions:
                    logging.info(f"Unknown label: {label}")

        return JSONResponse({
            "description": desc,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def add_article(label: str) -> str:
    vowels = "aeiou"
    article = "an" if label[0].lower() in vowels else "a"
    return f"{article} {label}"

@app.post("/safety-alerts")
async def safety_alerts(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return detect_objects(image_bytes)

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = np.asarray(bytearray(contents), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        text = pytesseract.image_to_string(thresh)
        return {"text": text.strip().replace('\n', ' ')}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
