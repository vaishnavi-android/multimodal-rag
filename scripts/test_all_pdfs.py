from pathlib import Path

from src.ingestion.pdf_parser import parse_pdf


def find_pdfs(data_folder: Path):
    """
    Find all PDF files inside the data folder.
    """
    return list(data_folder.rglob("*.pdf"))


def main():

    print("=" * 70)
    print("MULTIMODAL RAG - ALL PDF PARSER TEST")
    print("=" * 70)

    data_folder = Path("data")

    if not data_folder.exists():
        print(f"\nERROR: Data folder not found: {data_folder}")
        return

    pdf_files = find_pdfs(data_folder)

    if not pdf_files:
        print("\nNo PDF files found.")
        return

    print(f"\nTotal PDFs found: {len(pdf_files)}")

    successful_pdfs = 0
    failed_pdfs = 0

    total_pages = 0
    total_text_pages = 0
    total_ocr_pages = 0
    total_pages_with_images = 0

    for index, pdf_path in enumerate(pdf_files, start=1):

        print("\n")
        print("=" * 70)
        print(f"PDF {index}/{len(pdf_files)}")
        print("=" * 70)

        print(f"File: {pdf_path}")

        try:
            parsed_document = parse_pdf(pdf_path)
            pages = parsed_document.pages

            pdf_text_pages = 0
            pdf_ocr_pages = 0
            pdf_pages_with_images = 0

            for page in pages:

                # OCR status
                if page.needs_ocr:
                    pdf_ocr_pages += 1
                    total_ocr_pages += 1
                else:
                    pdf_text_pages += 1
                    total_text_pages += 1

                # Embedded image status
                if page.has_images:
                    pdf_pages_with_images += 1
                    total_pages_with_images += 1

            total_pages += len(pages)

            print(f"Pages found              : {len(pages)}")
            print(f"Text-based pages         : {pdf_text_pages}")
            print(f"OCR-required pages       : {pdf_ocr_pages}")
            print(f"Pages with images        : {pdf_pages_with_images}")

            successful_pdfs += 1

            print("\nStatus: SUCCESS")

        except Exception as error:

            failed_pdfs += 1

            print("\nStatus: FAILED")
            print(f"Error: {error}")

    print("\n")
    print("=" * 70)
    print("FINAL PDF PARSER SUMMARY")
    print("=" * 70)

    print(f"Total PDFs tested        : {len(pdf_files)}")
    print(f"Successful PDFs          : {successful_pdfs}")
    print(f"Failed PDFs              : {failed_pdfs}")
    print(f"Total pages              : {total_pages}")
    print(f"Text-based pages         : {total_text_pages}")
    print(f"OCR-required pages       : {total_ocr_pages}")
    print(f"Pages containing images  : {total_pages_with_images}")

    print("=" * 70)


if __name__ == "__main__":
    main()