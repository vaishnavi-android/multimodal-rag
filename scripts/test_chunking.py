from pathlib import Path

from src.ingestion.pipeline import ingest_document
from src.preprocessing.preprocessing_pipeline import (
    preprocess_content_units,
)
from src.chunking.chunker import chunk_units


def main():
    file_path = Path(
        "data/bucket_2/doc2.pdf"
    )

    bucket_id = "bucket_2"

    print("=" * 70)
    print("CHUNKING DEBUG TEST")
    print("=" * 70)

    print("\n1. INGESTING DOCUMENT")
    print("-" * 70)

    units = ingest_document(
        file_path,
        bucket_id,
    )

    print(f"Raw units: {len(units)}")

    print("\n2. PREPROCESSING")
    print("-" * 70)

    processed_units = preprocess_content_units(
        units
    )

    print(
        f"Processed units: "
        f"{len(processed_units)}"
    )

    print("\n3. PREPROCESSED UNITS")
    print("=" * 70)

    for i, unit in enumerate(
        processed_units,
        start=1,
    ):
        print(f"\nUNIT {i}")
        print("-" * 70)
        print(
            f"Type       : {unit.content_type}"
        )
        print(
            f"Page       : {unit.page_number}"
        )
        print(
            f"Characters : {len(unit.text)}"
        )

        print("\nTEXT:")
        print(unit.text[:1000])

        if len(unit.text) > 1000:
            print("\n... [TRUNCATED] ...")

    print("\n4. CHUNKING")
    print("=" * 70)

    chunks = chunk_units(
        processed_units
    )

    print(
        f"\nTotal chunks created: "
        f"{len(chunks)}"
    )

    for i, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(f"\nCHUNK {i}")
        print("-" * 70)

        print(
            f"Type       : {chunk.content_type}"
        )

        print(
            f"Page       : {chunk.page_number}"
        )

        print(
            f"Characters : {len(chunk.text)}"
        )

        print("\nTEXT:")
        print(chunk.text[:1000])

        if len(chunk.text) > 1000:
            print("\n... [TRUNCATED] ...")

    print("\n" + "=" * 70)
    print("CHUNKING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()