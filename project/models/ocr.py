from PIL import Image
import pytesseract
import os

# --- Configure Tesseract Path (if needed) ---
# Check common paths or allow environment variable override
TESSERACT_PATH_ENV = os.getenv("TESSERACT_CMD")
tesseract_cmd_path = None

if TESSERACT_PATH_ENV and os.path.exists(TESSERACT_PATH_ENV):
     tesseract_cmd_path = TESSERACT_PATH_ENV
elif os.path.exists('/usr/bin/tesseract'): # Linux common path
     tesseract_cmd_path = '/usr/bin/tesseract'
elif os.path.exists('/usr/local/bin/tesseract'): # Another Linux common path
     tesseract_cmd_path = '/usr/local/bin/tesseract'
elif os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'): # Windows common path
     tesseract_cmd_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if tesseract_cmd_path:
     print(f"Using Tesseract at: {tesseract_cmd_path}")
     pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
else:
     print("Warning: Tesseract command path not explicitly set. Relying on system PATH.")
     # PyTesseract will try to find it in PATH if not set.

# --- OCR Function ---
def extract_text_from_image(image: Image.Image, lang: str = 'eng') -> str:
    """
    Extracts text from a PIL image using Tesseract.
    Supports specified language (default: English).
    """
    print(f"Performing OCR with language: {lang}...")
    try:
        # Consider adding image preprocessing here for better results:
        # - Convert to grayscale: image.convert('L')
        # - Increase contrast
        # - Apply thresholding (e.g., Otsu's method with OpenCV)
        # For now, use the raw image
        text = pytesseract.image_to_string(image, lang=lang, config='--psm 3') # PSM 3 is default auto page segmentation
        print(f"OCR Result Length: {len(text)}")
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        error_msg = ("ERROR: Tesseract is not installed or not found in system PATH. "
                     "Install Tesseract OCR and ensure it's discoverable, "
                     "or set the TESSERACT_CMD environment variable.")
        print(error_msg)
        # Re-raise as a runtime error so the API can catch it
        raise RuntimeError(error_msg)
    except Exception as e:
        print(f"Error during OCR: {e}")
        # Optionally, check for specific pytesseract errors if needed
        return "" # Return empty string on other errors