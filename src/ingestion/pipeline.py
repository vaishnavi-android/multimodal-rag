from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from src.ingestion.file_detector import detect_file_type, FileType
from src.ingestion.pdf_parser import parse_pdf, render_page_to_image
from src.ingestion.image_parser import parse_image
from src.ingestion.ocr import run_ocr
from src.ingestion.table_parser import extract_tables
from src.preprocessing.ocr_cleaner import clean_ocr_text
from src.ingestion.image_filter import should_attempt_ocr


@dataclass
class RawContentUnit:
    """Represents extracted content before preprocessing and chunking."""

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
    Detect the file type and send it to the appropriate ingestion pipeline.
    """

    file_path = Path(file_path)
    file_type = detect_file_type(file_path)

    if file_type == FileType.PDF:
        return _ingest_pdf(file_path, bucket_id)

    if file_type == FileType.IMAGE:
        return _ingest_image(file_path, bucket_id)

    raise ValueError(
        f"Unsupported file type for {file_path}"
    )


def _ingest_pdf(
    file_path: Path,
    bucket_id: str,
) -> List[RawContentUnit]:
    """
    Extract text, tables, and useful embedded-image OCR content from a PDF.
    """

    parsed = parse_pdf(file_path)

    units: List[RawContentUnit] = []

    # Keeps track of pages already processed using full-page OCR.
    scanned_pages = set()

    # --------------------------------------------------
    # 1. PROCESS EACH PDF PAGE
    # --------------------------------------------------

    for page in parsed.pages:

        text = page.text

        # If the page has little/no extractable text,
        # render the complete page and run OCR.
        if page.needs_ocr:

            rendered = render_page_to_image(
                file_path,
                page.page_number,
            )

            text = run_ocr(rendered)
            text = clean_ocr_text(text)

            scanned_pages.add(
                page.page_number
            )

        # Add normal/OCR page text.
        if text and text.strip():

            units.append(
                RawContentUnit(
                    document_path=file_path,
                    bucket_id=bucket_id,
                    content_type="text",
                    text=text.strip(),
                    page_number=page.page_number,
                )
            )

        # --------------------------------------------------
        # 2. EXTRACT TABLES FROM THIS PAGE
        # --------------------------------------------------

        tables = extract_tables(
            file_path,
            page.page_number,
        )

        for table_md in tables:

            if not table_md or not table_md.strip():
                continue

            units.append(
                RawContentUnit(
                    document_path=file_path,
                    bucket_id=bucket_id,
                    content_type="table",
                    text=table_md.strip(),
                    page_number=page.page_number,
                )
            )

    # --------------------------------------------------
    # 3. PROCESS EMBEDDED IMAGES
    # --------------------------------------------------

    for embedded in parsed.embedded_images:

        # If the complete page was already OCR'd,
        # don't OCR its embedded images again.
        if embedded.page_number in scanned_pages:
            continue

        # Skip small or visually unimportant images.
        if not should_attempt_ocr(embedded.image):
            continue

        image_text = run_ocr(
            embedded.image
        )

        image_text = clean_ocr_text(
            image_text
        )

        # Skip images where OCR found nothing.
        if not image_text or not image_text.strip():
            continue

        units.append(
            RawContentUnit(
                document_path=file_path,
                bucket_id=bucket_id,
                content_type="image_ocr",
                text=image_text.strip(),
                page_number=embedded.page_number,
            )
        )

    # --------------------------------------------------
    # IMPORTANT: RETURN THE UNITS
    # --------------------------------------------------

    return units


def _ingest_image(
    file_path: Path,
    bucket_id: str,
) -> List[RawContentUnit]:
    """
    Extract OCR text from a standalone image.
    """

    parsed = parse_image(file_path)

    if not parsed.ocr_text or not parsed.ocr_text.strip():
        return []

    return [
        RawContentUnit(
            document_path=file_path,
            bucket_id=bucket_id,
            content_type="image_description",
            text=parsed.ocr_text.strip(),
            page_number=None,
        )
    ]