from pathlib import Path

from src.ingestion.table_parser import extract_tables


def main():
    print("=" * 70)
    print("TABLE PARSER TEST")
    print("=" * 70)

    pdf_path = Path("data/bucket_1/doc1.pdf")

    print(f"\nFile: {pdf_path}")

    if not pdf_path.exists():
        print("\nERROR: PDF file not found")
        return

    # Change this page number to the page containing your tables
    page_number = 1

    print(f"Page: {page_number}")
    print("\nExtracting tables...\n")

    tables = extract_tables(
        file_path=pdf_path,
        page_number=page_number,
    )

    print("=" * 70)
    print(f"TABLES FOUND: {len(tables)}")
    print("=" * 70)

    for index, table in enumerate(tables, start=1):
        print(f"\nTABLE {index}")
        print("-" * 70)
        print(table)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()