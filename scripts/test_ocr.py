from pathlib import Path

from PIL import Image

from src.ingestion.ocr import run_ocr


IMAGE_PATH = Path("data/bucket_1/images.jpg")


def main():
    print("=" * 70)
    print("RAPIDOCR TEST")
    print("=" * 70)

    print(f"\nImage path: {IMAGE_PATH.resolve()}")
    print(f"Exists: {IMAGE_PATH.exists()}")

    if not IMAGE_PATH.exists():
        print(f"\nERROR: Image not found: {IMAGE_PATH}")
        return

    with Image.open(IMAGE_PATH) as image:
        image = image.convert("RGB")
        text = run_ocr(image)

    print("\nEXTRACTED TEXT")
    print("-" * 70)
    print(text)

    print("\nCharacters extracted:", len(text))

    print("\n" + "=" * 70)
    print("RAPIDOCR TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()