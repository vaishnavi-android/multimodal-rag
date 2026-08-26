import re


def clean_ocr_text(text: str) -> str:
    """Apply safe OCR text normalization."""

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ ]{2,}", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines).strip()