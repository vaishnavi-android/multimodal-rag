from pathlib import Path

from src.ingestion.pdf_parser import parse_pdf
from src.preprocessing.cleaner import clean_text


PDF_PATH = Path("data/bucket_1/doc1.pdf")


def main():
    print("=" * 70)
    print("TEXT CLEANER TEST")
    print("=" * 70)

    parsed_pdf = parse_pdf(PDF_PATH)

    for page in parsed_pdf.pages:
        original = page.text
        cleaned = clean_text(original)

        print("\n" + "=" * 70)
        print(f"PAGE {page.page_number}")
        print("=" * 70)

        print("\n--- ORIGINAL ---")
        print(original[:1500])

        print("\n--- CLEANED ---")
        print(cleaned[:1500])

        print("\nOriginal characters :", len(original))
        print("Cleaned characters  :", len(cleaned))

    print("\n" + "=" * 70)
    print("TEXT CLEANER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()