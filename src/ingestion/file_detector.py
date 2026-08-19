"""
Detects file type for a given document path so the ingestion pipeline
can route it to the correct parser (PDF vs. image, etc.).
"""

from pathlib import Path
from enum import Enum


class FileType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PDF_EXTENSIONS = {".pdf"}


def detect_file_type(file_path: Path) -> FileType:
    """Return the FileType for a given file, based on its extension.

    NOTE: extension-based detection is a reasonable first pass. If you hit
    mislabeled files later, swap this for content-sniffing (e.g. the
    `filetype` or `python-magic` library) without touching any caller.
    """
    ext = Path(file_path).suffix.lower()

    if ext in PDF_EXTENSIONS:
        return FileType.PDF
    if ext in IMAGE_EXTENSIONS:
        return FileType.IMAGE
    return FileType.UNKNOWN


def list_bucket_documents(bucket_dir: Path):
    """Return every supported document inside a bucket directory."""
    bucket_dir = Path(bucket_dir)
    docs = []
    for path in sorted(bucket_dir.iterdir()):
        if path.is_file() and detect_file_type(path) != FileType.UNKNOWN:
            docs.append(path)
    return docs
