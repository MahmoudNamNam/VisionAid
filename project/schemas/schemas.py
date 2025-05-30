from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Input Schemas ---
class SettingsInput(BaseModel):
    interested_categories: Optional[List[str]] = None
    language: Optional[str] = Field(None, pattern="^(en)$")

# --- Output Schemas ---
class DetectedObject(BaseModel):
    label: str
    confidence: float
    box: List[int] # [xmin, ymin, xmax, ymax]

class ObjectDetectionResponse(BaseModel):
    description: str
    detected_objects: List[DetectedObject]
    hazards_detected: List[DetectedObject]
    alerts: List[str]

class SceneResponse(BaseModel):
    description: str
    scene_label: str
    confidence: float

class OCRResponse(BaseModel):
    extracted_text: str
    language: str = "en"

class SettingsResponse(BaseModel):
    interested_categories: List[str]
    language: str