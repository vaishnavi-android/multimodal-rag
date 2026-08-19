from typing import List

from src.ingestion.pipeline import RawContentUnit
from src.preprocessing.cleaner import clean_text
from src.preprocessing.ocr_cleaner import clean_ocr_text
from src.preprocessing.document_deduplicator import (
    DeduplicatedBlock,
    deduplicate_document_text,
)


def preprocess_content_units(
    units: List[RawContentUnit],
) -> List[RawContentUnit]:
    """
    Preprocess all RawContentUnit objects belonging to a document.

    Text:
        clean -> OCR correction -> logical blocks -> document-level
        deduplication.

    Tables:
        preserve as complete structured units.
        Tables are NOT deduplicated at this stage.

    Image descriptions:
        clean -> OCR correction -> preserve as complete units.
        Image descriptions are NOT deduplicated at this stage.

    The purpose of this stage is to clean and deduplicate textual
    content while preserving the original occurrence and provenance
    of tables and image-derived content.
    """

    processed: List[RawContentUnit] = []

    # Document-wide text blocks used by logical-block
    # deduplication.
    existing_text_blocks: List[DeduplicatedBlock] = []

    for unit in units:

        if not unit.text or not unit.text.strip():
            continue

        # =========================================================
        # TEXT
        # =========================================================

        if unit.content_type == "text":

            cleaned = clean_text(unit.text)
            cleaned = clean_ocr_text(cleaned)

            if not cleaned.strip():
                continue

            new_blocks = deduplicate_document_text(
                page_number=unit.page_number or 0,
                text=cleaned,
                existing_blocks=existing_text_blocks,
            )

            for block in new_blocks:

                new_unit = RawContentUnit(
                    document_path=unit.document_path,
                    bucket_id=unit.bucket_id,
                    content_type="text",
                    text=block.content,
                    page_number=unit.page_number,
                )

                processed.append(new_unit)
                existing_text_blocks.append(block)

            continue

        # =========================================================
        # TABLE
        # =========================================================

        if unit.content_type == "table":

            # Tables must remain complete structured units.
            #
            # IMPORTANT:
            # Do NOT compare tables for document-level duplicates.
            # Two identical tables on different pages are still two
            # source occurrences and must retain their provenance.

            table_text = unit.text.strip()

            if not table_text:
                continue

            table_unit = RawContentUnit(
                document_path=unit.document_path,
                bucket_id=unit.bucket_id,
                content_type="table",
                text=table_text,
                page_number=unit.page_number,
            )

            processed.append(table_unit)

            continue

        # =========================================================
        # IMAGE DESCRIPTION
        # =========================================================

        if unit.content_type == "image_description":

            # Image descriptions remain complete units.
            #
            # Do not perform document-level deduplication here.
            # Multiple images may legitimately have identical or
            # very similar descriptions while representing different
            # source locations.

            cleaned = clean_text(unit.text)
            cleaned = clean_ocr_text(cleaned)

            if not cleaned.strip():
                continue

            image_unit = RawContentUnit(
                document_path=unit.document_path,
                bucket_id=unit.bucket_id,
                content_type="image_description",
                text=cleaned,
                page_number=unit.page_number,
            )

            processed.append(image_unit)

            continue

        # =========================================================
        # UNKNOWN CONTENT TYPE
        # =========================================================

        # Preserve unknown future multimodal content types rather
        # than silently deleting them.
        processed.append(unit)

    return processed