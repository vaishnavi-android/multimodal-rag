"""
Parses standalone image files (PNG/JPG/JPEG).

The image is opened and OCR is used to extract text.
The rest of the RAG pipeline receives the extracted text.
"""

from pathlib import Path
from dataclasses import dataclass

from PIL import Image

from src.ingestion.ocr import run_ocr


@dataclass
class ParsedImage:
    """Text extracted from a standalone image."""

    file_path: Path
    text: str


def parse_image(file_path: Path) -> ParsedImage:
    """Extract text from a standalone image."""

    file_path = Path(file_path)

    with Image.open(file_path) as image:
        image = image.convert("RGB")
        text = run_ocr(image)

    return ParsedImage(
        file_path=file_path,
        text=text,
    )