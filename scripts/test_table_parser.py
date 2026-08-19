from pathlib import Path

from src.ingestion.table_parser import extract_tables


PDF_PATH = Path("data/bucket_1/doc1.pdf")


def main():
    print("=" * 70)
    print("TABLE PARSER TEST")
    print("=" * 70)

    # Page 2 contains the test table.
    tables = extract_tables(PDF_PATH, page_number=2)

    print(f"\nTables found: {len(tables)}")

    for index, table in enumerate(tables, start=1):
        print("\n" + "-" * 70)
        print(f"TABLE {index}")
        print("-" * 70)
        print(table)

    print("\n" + "=" * 70)
    print("TABLE PARSER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()