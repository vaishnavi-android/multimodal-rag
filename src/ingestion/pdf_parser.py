"""
PDF parsing.

Extracts per-page text using PyMuPDF (fitz). Pages with little/no
extractable text are flagged as likely scanned so the caller can route
them through OCR (see ocr.py).

Also extracts embedded images (charts, diagrams, photos) from each page
as PIL Images so they can be processed with OCR rather than being
detected and skipped.

Table extraction is intentionally kept in table_parser.py so this module
stays focused on text, page structure, and embedded images.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import hashlib
import io


# Minimum characters of extracted text before we trust a page is
# text-based rather than a scanned image.
MIN_TEXT_CHARS_PER_PAGE = 20

# Skip tiny embedded images (icons, bullet dots, logos, decorative
# rules) - these aren't meaningful charts/diagrams and just add noise.
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100


@dataclass
class EmbeddedImage:
    page_number: int
    image_index: int
    image: object
    image_hash: str


@dataclass
class PageContent:
    page_number: int
    text: str
    needs_ocr: bool
    has_images: bool = False


@dataclass
class ParsedPDF:
    file_path: Path
    pages: List[PageContent] = field(default_factory=list)
    embedded_images: List[EmbeddedImage] = field(default_factory=list)


def parse_pdf(file_path: Path) -> ParsedPDF:
    """
    Parse a PDF into per-page text content plus unique embedded images.
    """

    import fitz
    from PIL import Image

    file_path = Path(file_path)
    doc = fitz.open(file_path)

    pages: List[PageContent] = []
    embedded_images: List[EmbeddedImage] = []
    seen_images = set()

    try:
        for i, page in enumerate(doc):
            page_number = i + 1

            text = page.get_text("text") or ""
            image_refs = page.get_images(full=True)

            pages.append(
                PageContent(
                    page_number=page_number,
                    text=text.strip(),
                    needs_ocr=(
                        len(text.strip()) < MIN_TEXT_CHARS_PER_PAGE
                    ),
                    has_images=len(image_refs) > 0,
                )
            )

            for img_index, img_ref in enumerate(image_refs):
                xref = img_ref[0]

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    image_hash = hashlib.md5(
                        image_bytes
                    ).hexdigest()

                    # Skip exact duplicate embedded images.
                    if image_hash in seen_images:
                        continue

                    pil_image = Image.open(
                        io.BytesIO(image_bytes)
                    ).convert("RGB")

                except Exception:
                    # Corrupt or unsupported image should never
                    # break the complete document ingestion.
                    continue

                # Skip tiny decorative images.
                if (
                    pil_image.width < MIN_IMAGE_WIDTH
                    or pil_image.height < MIN_IMAGE_HEIGHT
                ):
                    continue

                seen_images.add(image_hash)

                embedded_images.append(
                    EmbeddedImage(
                        page_number=page_number,
                        image_index=img_index,
                        image=pil_image,
                        image_hash=image_hash,
                    )
                )

    finally:
        doc.close()

    return ParsedPDF(
        file_path=file_path,
        pages=pages,
        embedded_images=embedded_images,
    )


def render_page_to_image(
    file_path: Path,
    page_number: int,
    dpi: int = 200,
):
    """
    Render a single PDF page to a PIL Image for OCR fallback.

    page_number is 1-indexed.
    """

    import fitz
    from PIL import Image

    doc = fitz.open(Path(file_path))

    try:
        page = doc[page_number - 1]

        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        img = Image.frombytes(
            "RGB",
            (pix.width, pix.height),
            pix.samples,
        )

        return img

    finally:
        doc.close()