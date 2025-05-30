import json
import aiofiles
from typing import Dict, Any, List
from schemas.schemas import SettingsResponse # Use the schema
import os

# --- File Paths ---
# Use absolute paths or paths relative to the project root
# Ensure these directories exist
DATA_DIR = "data"
SETTINGS_FILE = os.path.join(DATA_DIR, "user_settings.json")
HAZARD_FILE = os.path.join(DATA_DIR, "hazard_list.json")
COCO_LABELS_FILE = os.path.join(DATA_DIR, "coco_labels.json")
PLACES_LABELS_FILE = os.path.join(DATA_DIR, "places365_labels.txt")

DEFAULT_SETTINGS = {"interested_categories": [], "language": "en"}

async def read_settings() -> SettingsResponse:
    """Reads user settings asynchronously."""
    if not os.path.exists(SETTINGS_FILE):
        await write_settings(DEFAULT_SETTINGS)
        return SettingsResponse(**DEFAULT_SETTINGS)

    try:
        async with aiofiles.open(SETTINGS_FILE, mode='r') as f:
            content = await f.read()
            settings_data = json.loads(content)
            # Ensure all keys exist, falling back to defaults
            for key, value in DEFAULT_SETTINGS.items():
                settings_data.setdefault(key, value)
            return SettingsResponse(**settings_data)
    except json.JSONDecodeError:
        print(f"Warning: Error decoding {SETTINGS_FILE}. Using default settings.")
        await write_settings(DEFAULT_SETTINGS) # Overwrite corrupted file
        return SettingsResponse(**DEFAULT_SETTINGS)
    except Exception as e:
        print(f"Error reading settings: {e}. Using default settings.")
        return SettingsResponse(**DEFAULT_SETTINGS)


async def write_settings(settings_data: Dict[str, Any]) -> None:
    """Writes user settings asynchronously."""
    valid_data = {key: settings_data.get(key, DEFAULT_SETTINGS[key])
                  for key in DEFAULT_SETTINGS}
    try:
        os.makedirs(DATA_DIR, exist_ok=True) # Ensure data directory exists
        async with aiofiles.open(SETTINGS_FILE, mode='w') as f:
            await f.write(json.dumps(valid_data, indent=4))
    except Exception as e:
        print(f"Error writing settings: {e}")


async def get_list_from_json(file_path: str) -> List[str]:
    """Reads a list of strings from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}. Returning empty list.")
        return []
    try:
        async with aiofiles.open(file_path, mode='r') as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        print(f"Error reading JSON list from {file_path}: {e}. Returning empty list.")
        return []

async def get_list_from_txt(file_path: str) -> List[str]:
    """Reads a list of strings from a text file (one item per line)."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}. Returning empty list.")
        return []
    try:
        async with aiofiles.open(file_path, mode='r') as f:
            lines = await f.readlines()
            return [line.strip() for line in lines if line.strip()]
    except Exception as e:
        print(f"Error reading text list from {file_path}: {e}. Returning empty list.")
        return []

# Specific getters using the generic functions
async def get_hazard_list() -> List[str]:
    return await get_list_from_json(HAZARD_FILE)

async def get_coco_labels() -> List[str]:
    return await get_list_from_json(COCO_LABELS_FILE)

async def get_places_labels() -> List[str]:
    return await get_list_from_txt(PLACES_LABELS_FILE)