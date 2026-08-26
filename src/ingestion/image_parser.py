from pathlib import Path
from dataclasses import dataclass

from PIL import Image, ImageOps

from src.ingestion.ocr import run_ocr


@dataclass
class ParsedImage:
    file_path: Path
    ocr_text: str

    @property
    def combined_text(self) -> str:
        return self.ocr_text.strip()


def _prepare_for_ocr(image: Image.Image) -> Image.Image:
    """Upscale small images before OCR."""

    image = image.convert("RGB")

    width, height = image.size

    if width < 1000:
        scale = 1000 / width

        image = image.resize(
            (
                int(width * scale),
                int(height * scale),
            ),
            Image.Resampling.LANCZOS,
        )

    image = ImageOps.grayscale(image)

    return image


def parse_image(file_path: Path) -> ParsedImage:
    file_path = Path(file_path)

    with Image.open(file_path) as image:
        img = image.convert("RGB")

    print(f"[image_parser] Processing: {file_path.name}")
    print(f"[image_parser] Original size: {img.size}")

    prepared_image = _prepare_for_ocr(img)

    print(
        f"[image_parser] OCR size: "
        f"{prepared_image.size}"
    )

    ocr_text = run_ocr(prepared_image)

    print(
        f"[image_parser] OCR characters extracted: "
        f"{len(ocr_text)}"
    )

    return ParsedImage(
        file_path=file_path,
        ocr_text=ocr_text,
    )