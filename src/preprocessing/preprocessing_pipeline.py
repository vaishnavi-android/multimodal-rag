from typing import List

from src.ingestion.pipeline import RawContentUnit
from src.preprocessing.cleaner import clean_text
from src.preprocessing.deduplicator import deduplicate_document_text


def preprocess_content_units(
    units: List[RawContentUnit],
) -> List[RawContentUnit]:
    """
    Preprocess all RawContentUnit objects.

    Text:
        clean -> split into logical blocks -> document-level deduplication

    Tables:
        preserve as complete structured units.

    Image content:
        clean and preserve as complete units.
    """

    processed: List[RawContentUnit] = []

    # Stores text blocks already accepted from this document.
    existing_texts: List[str] = []

    for unit in units:

        if not unit.text or not unit.text.strip():
            continue

        # =========================================================
        # TEXT
        # =========================================================

        if unit.content_type == "text":

            cleaned = clean_text(unit.text)

            if not cleaned:
                continue

            # Split into logical blocks and remove duplicates
            # already seen in this document.
            new_blocks = deduplicate_document_text(
                text=cleaned,
                existing_texts=existing_texts,
            )

            for block in new_blocks:

                processed.append(
                    RawContentUnit(
                        document_path=unit.document_path,
                        bucket_id=unit.bucket_id,
                        content_type="text",
                        text=block,
                        page_number=unit.page_number,
                    )
                )

                # Add accepted block so future pages can be
                # compared against it.
                existing_texts.append(block)

            continue

        # =========================================================
        # TABLE
        # =========================================================

        if unit.content_type == "table":

            table_text = unit.text.strip()

            if table_text:

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

        # =========================================================
        # IMAGE CONTENT
        # =========================================================

        if unit.content_type == "image_description":

            cleaned = clean_text(unit.text)

            if cleaned:

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

        # Preserve unknown future content types.
        processed.append(unit)

    return processed