from pathlib import Path

from src.ingestion.pipeline import ingest_document
from src.preprocessing.preprocessing_pipeline import (
    preprocess_content_units,
)


DATA_DIR = Path("data")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
}


def get_documents():
    """
    Find all supported documents inside every bucket directory.

    Returns tuples of:
        (file_path, bucket_id)
    """

    documents = []

    for bucket_dir in sorted(DATA_DIR.iterdir()):
        if not bucket_dir.is_dir():
            continue

        bucket_id = bucket_dir.name

        for file_path in sorted(bucket_dir.iterdir()):
            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                documents.append(
                    (file_path, bucket_id)
                )

    return documents


def main():
    print("=" * 80)
    print("ALL DOCUMENT INGESTION + PREPROCESSING TEST")
    print("=" * 80)

    documents = get_documents()

    if not documents:
        print("\nNo supported documents found.")
        return

    print(f"\nDocuments found: {len(documents)}\n")

    passed = []
    failed = []

    total_raw_units = 0
    total_processed_units = 0

    total_text_units = 0
    total_table_units = 0
    total_image_units = 0

    for index, (file_path, bucket_id) in enumerate(
        documents,
        start=1,
    ):
        print("-" * 80)

        print(
            f"[{index}/{len(documents)}] "
            f"{file_path.name}"
        )

        print(f"Bucket: {bucket_id}")

        try:
            # ------------------------------------------------------
            # 1. REAL INGESTION PIPELINE
            # ------------------------------------------------------

            units = ingest_document(
                file_path=file_path,
                bucket_id=bucket_id,
            )

            total_raw_units += len(units)

            # ------------------------------------------------------
            # 2. REAL PREPROCESSING PIPELINE
            # ------------------------------------------------------

            processed_units = preprocess_content_units(
                units
            )

            total_processed_units += len(
                processed_units
            )

            # ------------------------------------------------------
            # 3. COUNT CONTENT TYPES
            # ------------------------------------------------------

            text_units = sum(
                1
                for unit in processed_units
                if unit.content_type == "text"
            )

            table_units = sum(
                1
                for unit in processed_units
                if unit.content_type == "table"
            )

            image_units = sum(
                1
                for unit in processed_units
                if unit.content_type == "image_description"
            )

            total_text_units += text_units
            total_table_units += table_units
            total_image_units += image_units

            # ------------------------------------------------------
            # RESULT
            # ------------------------------------------------------

            print("Status: PASS")
            print(f"RawContentUnits: {len(units)}")
            print(
                f"Processed units: "
                f"{len(processed_units)}"
            )
            print(f"Text units: {text_units}")
            print(f"Table units: {table_units}")
            print(f"Image units: {image_units}")

            passed.append(file_path)

        except Exception as error:
            print("Status: FAIL")

            print(
                f"Error: "
                f"{type(error).__name__}: {error}"
            )

            failed.append(
                (file_path, error)
            )

    # --------------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\nTotal documents: {len(documents)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")

    print(f"\nTotal RawContentUnits: {total_raw_units}")
    print(
        f"Total Processed Units: "
        f"{total_processed_units}"
    )

    print(f"\nTotal Text Units: {total_text_units}")
    print(f"Total Table Units: {total_table_units}")
    print(f"Total Image Units: {total_image_units}")

    # --------------------------------------------------------------
    # FAILED DOCUMENTS
    # --------------------------------------------------------------

    if failed:
        print("\nFAILED DOCUMENTS")
        print("-" * 80)

        for file_path, error in failed:
            print(
                f"{file_path} "
                f"-> {type(error).__name__}: {error}"
            )

    # --------------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------------

    print("\n" + "=" * 80)

    if failed:
        print("TEST COMPLETED WITH FAILURES")
    else:
        print(
            "ALL DOCUMENTS PASSED "
            "INGESTION + PREPROCESSING SUCCESSFULLY"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()