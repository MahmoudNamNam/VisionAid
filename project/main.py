from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time # For basic timing

from schemas import schemas
from utils import image_processing, text_generation, settings_manager
from models import object_detection, scene_description, ocr
from typing import List

# --- App Initialization ---
app = FastAPI(
    title="Vision Aid API",
    description="API endpoints for the Vision Aid application features. Uses PyTorch models.",
    version="1.1.0"
)

# --- CORS Middleware ---
origins = ["*"] # Allow all origins for development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State / Model Loading ---
app_state = {"models_loaded": False}

@app.on_event("startup")
async def startup_event():
    """Load models and initial settings when the app starts."""
    print("Application starting up...")
    start_time = time.time()

    # Load models (run blocking loads in executor)
    loop = asyncio.get_event_loop()
    print("Loading Object Detection Model...")
    await loop.run_in_executor(None, object_detection.load_model)

    print("Loading Scene Description Model...")
    await loop.run_in_executor(None, scene_description.load_model)
    # Note: OCR doesn't require explicit model loading here (Tesseract is called directly)

    # Load hazard list and pass it to the object detection module
    print("Loading Hazard List...")
    hazards = await settings_manager.get_hazard_list()
    # Run set_hazard_list in executor if it becomes complex, ok sync for now
    object_detection.set_hazard_list(hazards)

    # Load initial settings
    print("Loading Initial Settings...")
    app_state["current_settings"] = await settings_manager.read_settings()

    app_state["models_loaded"] = True # Set flag indicating (attempted) load
    end_time = time.time()
    print(f"Application startup complete. Models loaded: {app_state['models_loaded']}. Time: {end_time - start_time:.2f}s")
    # Check model status within modules if needed
    if object_detection.model is None:
        print("WARNING: Object detection model failed to load.")
    if scene_description.scene_model is None:
        print("WARNING: Scene description model failed to load.")


# --- Helper Dependency for Settings ---
async def get_current_settings() -> schemas.SettingsResponse:
    if "current_settings" not in app_state:
         # Fallback if startup loading failed or called before startup finished
         print("Settings not found in app_state, reloading...")
         app_state["current_settings"] = await settings_manager.read_settings()
    return app_state["current_settings"]

# --- API Endpoints ---
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
import io


# Async image loader
async def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

@app.post("/detect/objects")
async def detect_objects(
    image_file: UploadFile = File(...), 
    categories: str = Form(None)
):
    try:
        # Read bytes from image
        image_bytes = await image_file.read()
        image = await load_image_from_bytes(image_bytes)

        # Log for debugging
        print(f"Image loaded: {image_file.filename}")
        print(f"Format: {image.format}, Mode: {image.mode}, Size: {image.size}")
        print(f"Categories: {categories}")

        # Dummy return for now
        return {
            "filename": image_file.filename,
            "detected_categories": categories or "all",
            "format": image.format,
            "mode": image.mode,
            "size": image.size
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during object detection. Error: {e}")


    except ValueError as e:
         raise HTTPException(status_code=400, detail=str(e)) # Image loading errors
    except Exception as e:
        print(f"Error in /detect/objects: {type(e).__name__} - {e}")
        # You might want more specific error handling for ML library exceptions
        raise HTTPException(status_code=500, detail="Internal server error during object detection.")


@app.post("/describe/scene", response_model=schemas.SceneResponse)
async def describe_scene_endpoint(
    image_file: UploadFile = File(...)
):
    """
    Provides a description of the overall scene from an uploaded image.
    May return mocked data if the scene model failed to load.
    """
    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type. Please upload an image.")
    # Allow mocked response if model isn't fully loaded
    # if scene_description.scene_model is None:
    #      raise HTTPException(status_code=503, detail="Scene description model is not available.")

    try:
        start_time = time.time()
        image_bytes = await image_file.read()
        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, image_processing.load_image_from_bytes, image_bytes)

        # Run ML inference in executor
        scene_label, confidence = await loop.run_in_executor(
            None, scene_description.describe_scene, image
        )

        # Generate description text (sync)
        description = text_generation.create_scene_description_text(scene_label, confidence)

        end_time = time.time()
        print(f"Scene description request processed in {end_time - start_time:.2f}s")

        return schemas.SceneResponse(
            description=description,
            scene_label=scene_label, # Return the raw label from model
            confidence=confidence
        )

    except ValueError as e:
         raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in /describe/scene: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail="Internal server error during scene description.")


@app.post("/recognize/text", response_model=schemas.OCRResponse)
async def recognize_text_endpoint(
    settings: schemas.SettingsResponse = Depends(get_current_settings),
    image_file: UploadFile = File(...)
):
    """
    Detects and extracts text (OCR) from an uploaded image using Tesseract.
    """
    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type. Please upload an image.")

    try:
        start_time = time.time()
        image_bytes = await image_file.read()
        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, image_processing.load_image_from_bytes, image_bytes)

        # Map language setting to Tesseract code ('en' -> 'eng')
        lang_code = 'eng' # Default
        if settings.language == 'en':
             lang_code = 'eng'
        # Add mappings for other languages if supported by Tesseract and your app

        # Run blocking Tesseract OCR in executor
        extracted_text = await loop.run_in_executor(
            None, ocr.extract_text_from_image, image, lang_code
        )

        end_time = time.time()
        print(f"OCR request processed in {end_time - start_time:.2f}s")

        return schemas.OCRResponse(
            extracted_text=extracted_text,
            language=settings.language
        )
    except ValueError as e:
         raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e: # Catch Tesseract not found error from ocr.py
         print(f"RuntimeError during OCR: {e}")
         raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Error in /recognize/text: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail="Internal server error during text recognition.")


@app.get("/settings", response_model=schemas.SettingsResponse)
async def get_settings_endpoint(
    settings: schemas.SettingsResponse = Depends(get_current_settings)
):
    """Gets the current user settings."""
    return settings

@app.put("/settings", status_code=status.HTTP_204_NO_CONTENT)
async def update_settings_endpoint(new_settings: schemas.SettingsInput):
    """Updates user settings."""
    try:
        current_settings = await settings_manager.read_settings()
        update_data = new_settings.dict(exclude_unset=True)
        updated_settings_data = current_settings.dict()
        updated_settings_data.update(update_data)

        await settings_manager.write_settings(updated_settings_data)
        # Update in-memory state
        app_state["current_settings"] = schemas.SettingsResponse(**updated_settings_data)
        print(f"Settings updated: {app_state['current_settings']}")
        # No content response needed for PUT success
    except Exception as e:
        print(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to update settings.")

@app.get("/status")
async def get_status():
    """Returns the loading status of models."""
    return {
        "models_loaded": app_state.get("models_loaded", False),
        "object_detection_ready": object_detection.model is not None,
        "scene_description_ready": scene_description.scene_model is not None and scene_description.scene_model != "MOCK_MODEL",
        "scene_description_mocked": scene_description.scene_model == "MOCK_MODEL",
        "ocr_ready": True # Assuming pytesseract is installed correctly
    }


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Vision Aid API! Check /docs for endpoints."}

# --- Run Command (in terminal) ---
# uvicorn main:app --reload --host 0.0.0.0 --port 8000