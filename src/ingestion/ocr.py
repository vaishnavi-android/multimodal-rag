"""
OCR wrapper. Kept as a thin, swappable layer around Tesseract so the
engine can be changed via config without touching callers.
"""

import pytesseract

from src.config.settings import OCR_LANGUAGE

# Windows: tell pytesseract exactly where Tesseract is installed.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def run_ocr(image) -> str:
    """Run OCR on a PIL Image and return extracted text."""
    text = pytesseract.image_to_string(image, lang=OCR_LANGUAGE)
    return text.strip()