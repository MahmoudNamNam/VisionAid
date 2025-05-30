from typing import List
from schemas.schemas import DetectedObject

def create_object_description(objects: List[DetectedObject]) -> str:
    """Generates a simple description from detected objects."""
    if not objects:
        return "I don't see any objects clearly."

    object_counts = {}
    for obj in objects:
        object_counts[obj.label] = object_counts.get(obj.label, 0) + 1

    parts = []
    for label, count in object_counts.items():
        # Simple pluralization (can be improved with libraries like 'inflect')
        plural = label + "s" if count > 1 and not label.endswith('s') else label
        num_str = str(count) if count > 1 else "a"
        # Handle vowels for 'a'/'an'
        if count == 1 and label[0].lower() in "aeiou":
            num_str = "an"

        parts.append(f"{num_str} {plural}")

    if not parts:
         return "I see something, but cannot identify specific objects."

    # Construct the sentence
    if len(parts) == 1:
        description = "I see " + parts[0]
    elif len(parts) == 2:
        description = "I see " + parts[0] + " and " + parts[1]
    else: # More than 2
        description = "I see " + ", ".join(parts[:-1]) + ", and " + parts[-1]

    description += "."
    return description

def create_hazard_alerts(hazards: List[DetectedObject]) -> List[str]:
    """Generates specific alerts for detected hazards."""
    alerts = []
    hazard_counts = {}
    for hazard in hazards:
         hazard_counts[hazard.label] = hazard_counts.get(hazard.label, 0) + 1

    for label, count in hazard_counts.items():
        plural = label if count == 1 else label + "s" # Basic plural
        alerts.append(f"Warning: {plural.capitalize()} detected!")
    return alerts

def create_scene_description_text(scene_label: str, confidence: float) -> str:
    """Generates text for the scene description."""
    # Clean up Places365 label format (e.g., '/a/abbey' -> 'abbey')
    cleaned_label = scene_label.split('/')[-1].replace('_', ' ')

    certainty = "likely" if confidence > 0.7 else "possibly"
    article = "a"
    # Basic vowel check for 'a'/'an'
    if cleaned_label and cleaned_label[0].lower() in "aeiou":
         article = "an"
    return f"You are {certainty} in {article} {cleaned_label} environment."