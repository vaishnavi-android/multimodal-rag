from pathlib import Path

from src.ingestion.pipeline import ingest_document
from src.preprocessing.preprocessing_pipeline import (
    preprocess_content_units,
)


PDF_PATH = Path("data/bucket_1/doc1.pdf")
BUCKET_ID = "bucket_1"


def main():

    print("=" * 70)
    print("END-TO-END INGESTION + PREPROCESSING PIPELINE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. INGEST
    # ---------------------------------------------------------

    print("\n[1] INGESTING DOCUMENT...")

    raw_units = ingest_document(
        file_path=PDF_PATH,
        bucket_id=BUCKET_ID,
    )

    print(f"Raw content units: {len(raw_units)}")

    # ---------------------------------------------------------
    # 2. PREPROCESS
    # ---------------------------------------------------------

    print("\n[2] PREPROCESSING CONTENT...")

    processed_units = preprocess_content_units(raw_units)

    print(f"Processed content units: {len(processed_units)}")

    # ---------------------------------------------------------
    # 3. DISPLAY RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROCESSED CONTENT UNITS")
    print("=" * 70)

    for index, unit in enumerate(processed_units, start=1):

        print(f"\nUNIT {index}")
        print("-" * 70)

        print(f"Document     : {unit.document_path.name}")
        print(f"Bucket       : {unit.bucket_id}")
        print(f"Content type : {unit.content_type}")
        print(f"Page         : {unit.page_number}")

        print("\nCONTENT:")
        print("-" * 70)
        print(unit.text)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    text_units = sum(
        1 for unit in processed_units
        if unit.content_type == "text"
    )

    table_units = sum(
        1 for unit in processed_units
        if unit.content_type == "table"
    )

    image_units = sum(
        1 for unit in processed_units
        if unit.content_type == "image_description"
    )

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    print(f"Raw units       : {len(raw_units)}")
    print(f"Processed units : {len(processed_units)}")
    print(f"Text units      : {text_units}")
    print(f"Table units     : {table_units}")
    print(f"Image units     : {image_units}")

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()