from pathlib import Path

from src.preprocessing.preprocessing_pipeline import preprocess_pdf


PDF_PATH = Path("data/bucket_1/doc1.pdf")


def main():
    print("=" * 70)
    print("END-TO-END PREPROCESSING PIPELINE TEST")
    print("=" * 70)

    document = preprocess_pdf(PDF_PATH)

    print(f"\nFile: {document.file_path}")
    print(f"Pages: {len(document.pages)}")

    print("\n" + "=" * 70)
    print("DEDUPLICATED TEXT BLOCKS")
    print("=" * 70)

    for block_number, block in enumerate(
        document.text_blocks,
        start=1,
    ):
        print(
            f"\nBLOCK {block_number} "
            f"— PAGE {block.page_number} "
            f"— TYPE {block.block_type}"
        )
        print("-" * 70)
        print(block.content)

    print("\n" + "=" * 70)
    print("EXTRACTED TABLES")
    print("=" * 70)

    if not document.tables:
        print("\nNo tables found.")
    else:
        for table_number, table in enumerate(
            document.tables,
            start=1,
        ):
            print(f"\nTABLE {table_number}")
            print("-" * 70)
            print(table)

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    print(
        f"\nDeduplicated text blocks: "
        f"{len(document.text_blocks)}"
    )

    print(
        f"Extracted tables: "
        f"{len(document.tables)}"
    )

    print("\n" + "=" * 70)
    print("END-TO-END PREPROCESSING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()