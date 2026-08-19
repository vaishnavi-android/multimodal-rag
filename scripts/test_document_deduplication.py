from pathlib import Path

from src.ingestion.pdf_parser import parse_pdf
from src.preprocessing.ocr_cleaner import clean_ocr_text
from src.preprocessing.document_deduplicator import (
    deduplicate_document_text,
)


PDF_PATH = Path("data/bucket_1/doc1.pdf")


def main():
    print("=" * 70)
    print("LOGICAL-BLOCK DOCUMENT DEDUPLICATION TEST")
    print("=" * 70)

    parsed = parse_pdf(PDF_PATH)

    # This list represents the document-wide collection of
    # blocks that have already been kept.
    deduplicated_blocks = []

    for page in parsed.pages:

        # Clean OCR errors before deduplication.
        cleaned = clean_ocr_text(page.text)

        # Split the current page into logical blocks and remove
        # blocks that already appeared earlier in the document.
        page_blocks = deduplicate_document_text(
            page_number=page.page_number,
            text=cleaned,
            existing_blocks=deduplicated_blocks,
        )

        # Add the newly accepted blocks to the document-wide
        # collection so later pages can be compared against them.
        deduplicated_blocks.extend(page_blocks)

        print(
            f"\nPage {page.page_number}: "
            f"{len(page_blocks)} unique blocks kept"
        )

    print("\n" + "=" * 70)
    print("DEDUPLICATION RESULT")
    print("=" * 70)

    print(
        f"\nTotal unique blocks: "
        f"{len(deduplicated_blocks)}"
    )

    print("\n" + "=" * 70)
    print("UNIQUE CONTENT")
    print("=" * 70)

    for i, block in enumerate(
        deduplicated_blocks,
        start=1,
    ):
        print(
            f"\nBLOCK {i} — "
            f"PAGE {block.page_number} — "
            f"TYPE {block.block_type}"
        )

        print("-" * 70)
        print(block.content)

    print("\n" + "=" * 70)
    print("LOGICAL-BLOCK DEDUPLICATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()