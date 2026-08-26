"""
OCR wrapper using RapidOCR.
"""

import re

from rapidocr import RapidOCR


_engine = RapidOCR()


def _is_meaningful_text(text: str) -> bool:
    """
    Filter out extremely short or noisy OCR results.

    Embedded images in PDFs may contain photographs, decorative
    elements, or diagrams with little/no readable text. These should
    not become RAG content.
    """

    cleaned = text.strip()

    if not cleaned:
        return False

    # Remove whitespace to measure actual content.
    compact = re.sub(r"\s+", "", cleaned)

    # Reject extremely short results like:
    # "M", "2", "83", "mr"
    if len(compact) < 5:
        return False

    # Count alphabetic characters.
    alpha_count = sum(char.isalpha() for char in cleaned)

    # Reject results with almost no alphabetic content.
    if alpha_count < 3:
        return False

    return True


def run_ocr(image) -> str:
    """
    Run OCR using RapidOCR and return meaningful extracted text.
    """

    result = _engine(image)

    if not result or not result.txts:
        return ""

    text = "\n".join(result.txts).strip()

    if not _is_meaningful_text(text):
        return ""

    return text