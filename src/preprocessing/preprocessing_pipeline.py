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
    """Clean and preprocess document content units."""

    processed: List[RawContentUnit] = []
    existing_text_blocks: List[DeduplicatedBlock] = []

    for unit in units:
        if not unit.text or not unit.text.strip():
            continue

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

        if unit.content_type == "table":
            # Preserve tables as complete structured units.
            table_text = unit.text.strip()

            if not table_text:
                continue

            processed.append(
                RawContentUnit(
                    document_path=unit.document_path,
                    bucket_id=unit.bucket_id,
                    content_type="table",
                    text=table_text,
                    page_number=unit.page_number,
                )
            )

            continue

        if unit.content_type == "image_description":
            cleaned = clean_text(unit.text)
            cleaned = clean_ocr_text(cleaned)

            if not cleaned.strip():
                continue

            processed.append(
                RawContentUnit(
                    document_path=unit.document_path,
                    bucket_id=unit.bucket_id,
                    content_type="image_description",
                    text=cleaned,
                    page_number=unit.page_number,
                )
            )

            continue

        # Preserve unsupported or future content types.
        processed.append(unit)

    return processed