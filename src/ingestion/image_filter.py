"""
Utility functions for filtering embedded PDF images before OCR.
"""

from PIL import Image, ImageStat


MIN_OCR_IMAGE_WIDTH = 200
MIN_OCR_IMAGE_HEIGHT = 100


def should_attempt_ocr(image: Image.Image) -> bool:
    """
    Decide whether an image is worth sending to OCR.

    This is a lightweight pre-filter intended to reduce unnecessary OCR
    calls for tiny, blank, decorative, or nearly uniform images.
    """

    if image is None:
        return False

    width, height = image.size

    # Skip very small images.
    if width < MIN_OCR_IMAGE_WIDTH:
        return False

    if height < MIN_OCR_IMAGE_HEIGHT:
        return False

    # Convert to grayscale for simple image analysis.
    gray = image.convert("L")

    # Resize for faster statistics.
    gray = gray.resize((100, 100))

    stat = ImageStat.Stat(gray)

    # Very low standard deviation usually means the image is nearly
    # uniform (blank/simple decorative background).
    stddev = stat.stddev[0]

    if stddev < 5:
        return False

    return True