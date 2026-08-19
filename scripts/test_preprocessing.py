from pathlib import Path

from src.ingestion.pdf_parser import parse_pdf
from src.preprocessing.cleaner import clean_pages


PDF_PATH = Path("data/bucket_1/doc1.pdf")


def main():
    print("=" * 70)
    print("PREPROCESSING TEST")
    print("=" * 70)

    parsed_pdf = parse_pdf(PDF_PATH)

    original_pages = [page.text for page in parsed_pdf.pages]

    cleaned_pages = clean_pages(original_pages)

    for page, cleaned in zip(parsed_pdf.pages, cleaned_pages):
        print("\n" + "=" * 70)
        print(f"PAGE {page.page_number}")
        print("=" * 70)

        print("\n--- ORIGINAL ---")
        print(page.text[:1500])

        print("\n--- CLEANED ---")
        print(cleaned[:1500])

        print("\nOriginal characters :", len(page.text))
        print("Cleaned characters  :", len(cleaned))

    print("\n" + "=" * 70)
    print("PREPROCESSING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()