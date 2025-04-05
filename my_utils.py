from gtts import gTTS
import base64
from io import BytesIO
import torch
from torchvision import models, transforms
from PIL import Image
import os

translations = {
    "person": "شخص",
    "car": "سيارة",
    "bus": "حافلة",
    "truck": "شاحنة",
    "motorcycle": "دراجة نارية",
    "bicycle": "دراجة",
    "chair": "كرسي",
    "couch": "أريكة",
    "bed": "سرير",
    "dining table": "طاولة طعام"
}

def generate_speech(objects, lang="en"):
    """
    Generates speech from a list of objects with proper separators.

    Args:
        objects (list): A list of object labels (strings).
        lang (str, optional): The language for speech generation ('en' or 'ar'). Defaults to "en".

    Returns:
        str: A base64 encoded string of the generated audio.
    """
    if not objects:
        if lang == "ar":
            sentence = "لم يتم العثور على أي شيء."
        else:
            sentence = "I didn't find anything."
    else:
        if lang == "ar":
            translated_objects = [translations.get(obj, obj) for obj in objects]
            if len(translated_objects) == 1:
                sentence = f"أرى {translated_objects[0]}"
            else:
                sentence = f"أرى {' و '.join(translated_objects[:-1])} و {translated_objects[-1]}"
        else:
            if len(objects) == 1:
                sentence = f"I see {objects[0]}"
            else:
                sentence = f"I see {' and '.join(objects[:-1])} and {objects[-1]}"

    try:
        tts = gTTS(sentence, lang=lang)
        audio_io = BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        return base64.b64encode(audio_io.read()).decode()
    except Exception as e:
        print(f"Error during speech generation: {e}")
        return ""



if __name__ == '__main__':
    # Example usage
    objects_en_single = ["person"]
    audio_en_single_base64 = generate_speech(objects_en_single, "en")
    print("English Audio (Single Object - Base64):", audio_en_single_base64[:50] + "...")

    objects_en_multiple = ["person", "car", "chair"]
    audio_en_multiple_base64 = generate_speech(objects_en_multiple, "en")
    print("English Audio (Multiple Objects - Base64):", audio_en_multiple_base64[:50] + "...")

    objects_ar_single = ["person"]
    audio_ar_single_base64 = generate_speech(objects_ar_single, "ar")
    print("Arabic Audio (Single Object - Base64):", audio_ar_single_base64[:50] + "...")

    objects_ar_multiple = ["person", "car", "chair"]
    audio_ar_multiple_base64 = generate_speech(objects_ar_multiple, "ar")
    print("Arabic Audio (Multiple Objects - Base64):", audio_ar_multiple_base64[:50] + "...")

    empty_objects = []
    audio_empty_en = generate_speech(empty_objects, "en")
    print("Empty English Audio (Base64):", audio_empty_en)

    audio_empty_ar = generate_speech(empty_objects, "ar")
    print("Empty Arabic Audio (Base64):", audio_empty_ar)