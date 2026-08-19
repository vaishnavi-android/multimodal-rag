"""
Parses standalone image files (PNG/JPEG): runs OCR to pull out any text,
and calls image_understanding for a visual description. Both get merged
into one text representation so the rest of the pipeline (chunking,
embeddings) never needs to know the content originated from an image.
"""

from pathlib import Path
from dataclasses import dataclass

from src.ingestion.ocr import run_ocr
from src.ingestion import image_understanding


@dataclass
class ParsedImage:
    file_path: Path
    ocr_text: str
    description: str

    @property
    def combined_text(self) -> str:
        parts = [p for p in (self.description, self.ocr_text) if p]
        return "\n\n".join(parts)


def parse_image(file_path: Path) -> ParsedImage:
    from PIL import Image

    file_path = Path(file_path)
    img = Image.open(file_path).convert("RGB")

    ocr_text = run_ocr(img)
    description = image_understanding.describe_image(img)

    return ParsedImage(file_path=file_path, ocr_text=ocr_text, description=description)
