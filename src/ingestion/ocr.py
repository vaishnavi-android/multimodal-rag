"""
OCR wrapper using RapidOCR.

Provides a single run_ocr() function for the ingestion pipeline.
"""

from rapidocr import RapidOCR


# Initialize the OCR engine only once.
ocr_engine = RapidOCR()


def run_ocr(image) -> str:
    """
    Run OCR on an image and return extracted text.

    Parameters
    ----------
    image:
        PIL Image or another image format supported by RapidOCR.

    Returns
    -------
    str:
        Extracted text.
    """

    try:
        # Run RapidOCR
        result = ocr_engine(image)

        # If OCR produced no result
        if result is None:
            return ""

        # RapidOCR newer versions return RapidOCROutput.
        # The actual OCR results are stored in result.txts.
        texts = result.txts

        if not texts:
            return ""

        # Clean individual text values
        cleaned_texts = []

        for text in texts:
            if text and isinstance(text, str):
                cleaned_texts.append(text.strip())

        return "\n".join(cleaned_texts).strip()

    except Exception as error:
        print(f"[RapidOCR] OCR failed: {error}")
        return ""