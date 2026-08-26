"""
Central ingestion pipeline.

Accepts a document path and bucket_id, then converts the document into
RawContentUnit objects ready for preprocessing.

Supported flow:

PDF
    -> extracted text
    -> OCR fallback for scanned pages
    -> table extraction
    -> OCR for embedded images

PNG / JPG / JPEG
    -> OCR

This is the only module responsible for deciding which parser handles
each supported file type.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.ingestion.file_detector import FileType, detect_file_type
from src.ingestion.image_parser import parse_image
from src.ingestion.ocr import run_ocr
from src.ingestion.pdf_parser import parse_pdf, render_page_to_image
from src.ingestion.table_parser import extract_tables


@dataclass
class RawContentUnit:
    """
    Raw content extracted from a document before preprocessing
    and chunking.
    """

    document_path: Path
    bucket_id: str
    content_type: str
    text: str
    page_number: Optional[int] = None


def ingest_document(
    file_path: Path,
    bucket_id: str,
) -> List[RawContentUnit]:
    """
    Ingest a single supported document and return raw content units.
    """

    file_path = Path(file_path)
    file_type = detect_file_type(file_path)

    if file_type == FileType.PDF:
        return _ingest_pdf(file_path, bucket_id)

    if file_type == FileType.IMAGE:
        return _ingest_image(file_path, bucket_id)

    raise ValueError(f"Unsupported file type: {file_path}")

def _ingest_pdf(
    file_path: Path,
    bucket_id: str,
) -> List[RawContentUnit]:
    """
    Extract text, tables, and OCR text from images in a PDF.
    """

    parsed = parse_pdf(file_path)
    units: List[RawContentUnit] = []

    # Pages that required full-page OCR.
    # Embedded images on these pages are skipped to avoid duplicate OCR.
    scanned_pages = set()

    for page in parsed.pages:
        text = page.text

        # OCR fallback only for pages with little or no extractable text.
        if page.needs_ocr:
            image = render_page_to_image(
                file_path,
                page.page_number,
            )

            text = run_ocr(image)
            scanned_pages.add(page.page_number)

        if text:
            units.append(
                RawContentUnit(
                    document_path=file_path,
                    bucket_id=bucket_id,
                    content_type="text",
                    text=text,
                    page_number=page.page_number,
                )
            )

        # Extract tables separately.
        tables = extract_tables(
            file_path,
            page.page_number,
        )

        for table_text in tables:
            units.append(
                RawContentUnit(
                    document_path=file_path,
                    bucket_id=bucket_id,
                    content_type="table",
                    text=table_text,
                    page_number=page.page_number,
                )
            )

    # Process embedded images.
    #
    # Only meaningful OCR results are added. Images without readable
    # text are ignored.
    for embedded in parsed.embedded_images:

        # Avoid duplicate OCR when the complete page was already OCRed.
        if embedded.page_number in scanned_pages:
            continue

        # Skip very small images.
        width, height = embedded.image.size

        if width < 200 or height < 100:
            continue

        text = run_ocr(embedded.image)

        # RapidOCR may find no text in photographs/illustrations.
        if not text or len(text.strip()) < 3:
            continue

        units.append(
            RawContentUnit(
                document_path=file_path,
                bucket_id=bucket_id,
                content_type="image_description",
                text=text.strip(),
                page_number=embedded.page_number,
            )
        )

    return units


def _ingest_image(
    file_path: Path,
    bucket_id: str,
) -> List[RawContentUnit]:
    """
    Extract OCR text from a standalone image.
    """

    parsed = parse_image(file_path)

    if not parsed.text:
        return []

    return [
        RawContentUnit(
            document_path=file_path,
            bucket_id=bucket_id,
            content_type="image_description",
            text=parsed.text,
            page_number=None,
        )
    ]