from pathlib import Path

from PIL import Image

from src.ingestion.ocr import run_ocr


def main():

    image_folder = Path(
        "data/bucket_1/ocr_test_images"
    )

    print("=" * 70)
    print("RAPIDOCR MULTIPLE IMAGE TEST")
    print("=" * 70)

    if not image_folder.exists():

        print(f"\nERROR: Folder not found: {image_folder}")
        return

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_files = [
        file
        for file in image_folder.iterdir()
        if file.suffix.lower() in supported_extensions
    ]

    if not image_files:

        print("\nNo supported images found.")
        return

    print(f"\nImages found: {len(image_files)}")

    for index, image_path in enumerate(image_files, start=1):

        print("\n")
        print("=" * 70)
        print(f"IMAGE {index}: {image_path.name}")
        print("=" * 70)

        try:

            image = Image.open(image_path)

            print(f"Image size: {image.size}")
            print("\nRunning RapidOCR...")

            text = run_ocr(image)

            print("\nEXTRACTED TEXT")
            print("-" * 70)

            if text:
                print(text)
            else:
                print("[NO TEXT DETECTED]")

            print("\nTOTAL CHARACTERS:", len(text))

        except Exception as error:

            print(
                f"\nERROR processing {image_path.name}: "
                f"{error}"
            )

    print("\n")
    print("=" * 70)
    print("RAPIDOCR TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()