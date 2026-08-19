# scripts/test_preprocessing_integration.py

"""
Phase 1 integration test.

Verifies that the REAL multimodal ingestion pipeline feeds
correctly into the new preprocessing pipeline.

No embeddings or vector database are used here.
"""

from pathlib import Path

from src.ingestion.pipeline import ingest_document
from src.preprocessing.preprocessing_pipeline import (
    preprocess_content_units,
)


PDF_PATH = Path(
    "data/bucket_1/doc1.pdf"
)

BUCKET_ID = "bucket_1"


def main():

    print("=" * 70)
    print("PHASE 1 — PREPROCESSING INTEGRATION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. RAW INGESTION
    # ---------------------------------------------------------

    raw_units = ingest_document(
        PDF_PATH,
        BUCKET_ID,
    )

    print(
        f"\nRaw content units: "
        f"{len(raw_units)}"
    )

    raw_counts = {}

    for unit in raw_units:
        raw_counts[unit.content_type] = (
            raw_counts.get(
                unit.content_type,
                0,
            )
            + 1
        )

    print("\nRAW CONTENT TYPES")

    for content_type, count in raw_counts.items():
        print(
            f"  {content_type}: {count}"
        )

    # ---------------------------------------------------------
    # 2. PREPROCESSING
    # ---------------------------------------------------------

    processed_units = preprocess_content_units(
        raw_units
    )

    print(
        f"\nProcessed content units: "
        f"{len(processed_units)}"
    )

    processed_counts = {}

    for unit in processed_units:
        processed_counts[unit.content_type] = (
            processed_counts.get(
                unit.content_type,
                0,
            )
            + 1
        )

    print("\nPROCESSED CONTENT TYPES")

    for content_type, count in processed_counts.items():
        print(
            f"  {content_type}: {count}"
        )

    # ---------------------------------------------------------
    # 3. SHOW PROCESSED CONTENT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROCESSED CONTENT")
    print("=" * 70)

    for index, unit in enumerate(
        processed_units,
        start=1,
    ):

        print(
            f"\nUNIT {index}"
            f" — PAGE {unit.page_number}"
            f" — TYPE {unit.content_type}"
        )

        print("-" * 70)
        print(unit.text)

    # ---------------------------------------------------------
    # 4. SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)

    print(
        f"\nRaw units: "
        f"{len(raw_units)}"
    )

    print(
        f"Processed units: "
        f"{len(processed_units)}"
    )

    print(
        "\nText units: "
        f"{processed_counts.get('text', 0)}"
    )

    print(
        "Table units: "
        f"{processed_counts.get('table', 0)}"
    )

    print(
        "Image-description units: "
        f"{processed_counts.get('image_description', 0)}"
    )

    print("\n" + "=" * 70)
    print("PHASE 1 PREPROCESSING INTEGRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()