from PIL import Image
from rapidocr import RapidOCR


IMAGE_PATH = "data\\bucket_1\images.jpg"


def main():
    engine = RapidOCR()

    result = engine(IMAGE_PATH)

    print(result)


if __name__ == "__main__":
    main()