from pathlib import Path

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.ocr import run_ocr


PDF_PATH = Path("data/bucket_1/hees203.pdf")


def main():
    parsed = parse_pdf(PDF_PATH)

    print("=" * 70)
    print("HEES203 EMBEDDED IMAGE OCR TEST")
    print("=" * 70)

    print(f"\nEmbedded images found: {len(parsed.embedded_images)}\n")

    successful = 0
    empty = 0

    for i, embedded in enumerate(parsed.embedded_images, start=1):

        width, height = embedded.image.size

        print("-" * 70)
        print(
            f"[{i}] Page {embedded.page_number} | "
            f"Image {embedded.image_index} | "
            f"{width}x{height}"
        )

        text = run_ocr(embedded.image)

        if text:
            successful += 1

            preview = text[:150].replace("\n", " ")
            print("OCR: SUCCESS")
            print(f"Text preview: {preview}")

        else:
            empty += 1
            print("OCR: NO TEXT DETECTED")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total images     : {len(parsed.embedded_images)}")
    print(f"OCR success      : {successful}")
    print(f"No text detected : {empty}")


if __name__ == "__main__":
    main()