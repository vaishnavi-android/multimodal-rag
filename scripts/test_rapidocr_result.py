from pathlib import Path

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.ocr import _engine


PDF_PATH = Path("data/bucket_1/hees203.pdf")


def main():

    parsed = parse_pdf(PDF_PATH)

    # Test image 7 from your output:
    # Page 7, Image 1
    image = parsed.embedded_images[14].image

    result = _engine(image)

    print("=" * 70)
    print("RAPIDOCR RESULT INSPECTION")
    print("=" * 70)

    print("\nResult object:")
    print(result)

    print("\nAvailable attributes:")
    print(dir(result))

    print("\nTexts:")
    print(getattr(result, "txts", None))

    print("\nScores:")
    print(getattr(result, "scores", None))

    print("\nBoxes:")
    print(getattr(result, "boxes", None))


if __name__ == "__main__":
    main()