"""
Ingestion pipeline: takes a single document path + its bucket_id and
returns a list of RawContentUnit objects - one per page (PDF text),
one per table, one per embedded PDF image, or one for a whole standalone
image file - ready to hand off to preprocessing.

This is intentionally the ONLY place that decides "PDF -> pdf_parser,
image -> image_parser". Nothing else in the codebase should branch on
file type.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from src.ingestion.file_detector import detect_file_type, FileType
from src.ingestion.pdf_parser import parse_pdf, render_page_to_image
from src.ingestion.image_parser import parse_image
from src.ingestion.ocr import run_ocr
from src.ingestion.table_parser import extract_tables
from src.ingestion.image_understanding import describe_image


@dataclass
class RawContentUnit:
    """One page of text, one table, one embedded/standalone image's worth
    of raw extracted content, before cleaning/chunking. This maps closely
    to the metadata schema the spec requires downstream.
    """
    document_path: Path
    bucket_id: str
    content_type: str        # "text" | "image_description" | "table"
    text: str
    page_number: Optional[int] = None


def ingest_document(file_path: Path, bucket_id: str) -> List[RawContentUnit]:
    file_path = Path(file_path)
    file_type = detect_file_type(file_path)

    if file_type == FileType.PDF:
        return _ingest_pdf(file_path, bucket_id)
    elif file_type == FileType.IMAGE:
        return _ingest_image(file_path, bucket_id)
    else:
        raise ValueError(f"Unsupported file type for {file_path}")


def _ingest_pdf(file_path: Path, bucket_id: str) -> List[RawContentUnit]:
    parsed = parse_pdf(file_path)
    units: List[RawContentUnit] = []

    # Track which pages were already fully OCR'd as a scanned page (i.e.
    # rendered whole-page-to-image -> OCR). A scanned page is very often
    # internally stored as ONE big embedded image covering the entire
    # page - so without this guard, that same image would ALSO get
    # picked up by parsed.embedded_images below and get OCR'd a second
    # time + sent to the vision model as if it were "a chart found on
    # the page," producing a duplicate, misleading chunk. Pages in this
    # set are considered fully handled by the text loop already.
    scanned_pages = set()

    for page in parsed.pages:
        text = page.text

        # Scanned page with no extractable text layer -> OCR fallback.
        if page.needs_ocr:
            rendered = render_page_to_image(file_path, page.page_number)
            text = run_ocr(rendered)
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

        # Tables get their own units so they can be chunked/embedded as
        # coherent blocks rather than mixed into surrounding prose.
        for table_md in extract_tables(file_path, page.page_number):
            units.append(
                RawContentUnit(
                    document_path=file_path,
                    bucket_id=bucket_id,
                    content_type="table",
                    text=table_md,
                    page_number=page.page_number,
                )
            )

    # Embedded images (charts, diagrams, photos) found on normal
    # (non-scanned) pages: OCR any text inside them + get a vision
    # description, and turn each into its own retrievable unit - same
    # treatment a standalone PNG/JPEG gets in _ingest_image below.
    for embedded in parsed.embedded_images:
        if embedded.page_number in scanned_pages:
            continue  # already fully captured by the whole-page OCR above

        combined = _describe_and_ocr(embedded.image)
        if not combined:
            continue

        units.append(
            RawContentUnit(
                document_path=file_path,
                bucket_id=bucket_id,
                content_type="image_description",
                text=combined,
                page_number=embedded.page_number,
            )
        )

    return units


def _ingest_image(file_path: Path, bucket_id: str) -> List[RawContentUnit]:
    parsed = parse_image(file_path)
    if not parsed.combined_text:
        return []

    return [
        RawContentUnit(
            document_path=file_path,
            bucket_id=bucket_id,
            content_type="image_description",
            text=parsed.combined_text,
            page_number=None,
        )
    ]


def _describe_and_ocr(pil_image) -> str:
    """Shared helper: OCR text + vision description for a single PIL image,
    merged into one text block. Used for both embedded PDF images and
    standalone image files, so both get identical treatment.
    """
    ocr_text = run_ocr(pil_image)
    description = describe_image(pil_image)

    parts = [p for p in (description, ocr_text) if p]
    return "\n\n".join(parts)
