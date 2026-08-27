from pathlib import Path
import logging

logging.getLogger("RapidOCR").setLevel(logging.ERROR)
logging.getLogger("rapidocr").setLevel(logging.ERROR)
from src.ingestion.pipeline import ingest_document


def main():

    file_path = Path(
        "data/bucket_1/hees203.pdf"
    )

    bucket_id = "bucket_1"

    print("=" * 70)
    print("INGESTION PIPELINE TEST")
    print("=" * 70)

    print(f"\nFile: {file_path}")
    print(f"Bucket: {bucket_id}")

    print("\nRunning ingestion pipeline...\n")

    units = ingest_document(
        file_path=file_path,
        bucket_id=bucket_id,
    )

    print("=" * 70)
    print("RAW CONTENT UNITS")
    print("=" * 70)

    print(f"\nTotal units created: {len(units)}\n")

    type_counts = {}

    for unit in units:

        type_counts[unit.content_type] = (
            type_counts.get(unit.content_type, 0) + 1
        )

    print("CONTENT TYPE SUMMARY")

    for content_type, count in type_counts.items():
        print(f"{content_type}: {count}")

    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT")
    print("=" * 70)

    for index, unit in enumerate(units[:10], start=1):

        print(f"\nUNIT {index}")
        print("-" * 70)

        print(f"Type : {unit.content_type}")
        print(f"Page : {unit.page_number}")
        print(f"Chars: {len(unit.text)}")

        preview = unit.text[:500]

        print("\nText:")
        print(preview)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()