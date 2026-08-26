from pathlib import Path

from src.ingestion.pdf_parser import parse_pdf


# Test PDF
PDF_PATH = Path("data/bucket_1/hees203.pdf")


def main():
    print("=" * 60)
    print("PDF PARSER TEST")
    print("=" * 60)

    if not PDF_PATH.exists():
        print(f"\nERROR: File not found: {PDF_PATH}")
        return

    print(f"\nFile: {PDF_PATH}")

    # Run the existing parser
    parsed_pdf = parse_pdf(PDF_PATH)

    # Basic information
    print(f"Pages found: {len(parsed_pdf.pages)}")
    print(f"Embedded images found: {len(parsed_pdf.embedded_images)}")

    # Page information
    print("\n" + "=" * 60)
    print("PAGE INFORMATION")
    print("=" * 60)

    for page in parsed_pdf.pages:
        print(f"\nPage {page.page_number}")
        print(f"  Needs OCR : {page.needs_ocr}")
        print(f"  Has images: {page.has_images}")
        print(f"  Text chars: {len(page.text)}")

        # Show only the first 500 characters
        if page.text:
            preview = page.text[:500].replace("\n", " ")
            print(f"  Preview   : {preview}")
        else:
            print("  Preview   : [No extractable text]")

    # Embedded image information
    if parsed_pdf.embedded_images:
        print("\n" + "=" * 60)
        print("EMBEDDED IMAGES")
        print("=" * 60)

        for image in parsed_pdf.embedded_images:
            print(
                f"Page {image.page_number}, "
                f"Image {image.image_index}, "
                f"Size: {image.image.width}x{image.image.height}"
            )

    print("\n" + "=" * 60)
    print("PDF PARSER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()