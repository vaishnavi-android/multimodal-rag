from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from src.ingestion.file_detector import detect_file_type, FileType
from src.ingestion.pdf_parser import parse_pdf, render_page_to_image
from src.ingestion.image_parser import parse_image
from src.ingestion.ocr import run_ocr
from src.ingestion.table_parser import extract_tables


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

    parsed = parse_pdf(file_path)
    units: List[RawContentUnit] = []

    scanned_pages = set()

    for page in parsed.pages:

        text = page.text

        # Use OCR when the PDF page has no extractable text.
        if page.needs_ocr:
            rendered = render_page_to_image(
                file_path,
                page.page_number,
            )

            text = run_ocr(rendered)

            scanned_pages.add(
                page.page_number
            )

        if text and text.strip():

            units.append(
                RawContentUnit(
                    document_path=file_path,
                    bucket_id=bucket_id,
                    content_type="text",
                    text=text,
                    page_number=page.page_number,
                )
            )

        # Keep tables as separate structured units.
        for table_md in extract_tables(
            file_path,
            page.page_number,
        ):

            units.append(
                RawContentUnit(
                    document_path=file_path,
                    bucket_id=bucket_id,
                    content_type="table",
                    text=table_md,
                    page_number=page.page_number,
                )
            )

    # OCR embedded images from normal PDF pages.
    for embedded in parsed.embedded_images:

        # Scanned pages were already processed using full-page OCR.
        if embedded.page_number in scanned_pages:
            continue

        image_text = run_ocr(
            embedded.image
        )

        if not image_text or not image_text.strip():
            continue

        units.append(
            RawContentUnit(
                document_path=file_path,
                bucket_id=bucket_id,
                content_type="image_description",
                text=image_text,
                page_number=embedded.page_number,
            )
        )

    return units


def _ingest_image(
    file_path: Path,
    bucket_id: str,
) -> List[RawContentUnit]:

    parsed = parse_image(file_path)

    if not parsed.ocr_text or not parsed.ocr_text.strip():
        return []

    return [
        RawContentUnit(
            document_path=file_path,
            bucket_id=bucket_id,
            content_type="image_description",
            text=parsed.ocr_text,
            page_number=None,
        )
    ]