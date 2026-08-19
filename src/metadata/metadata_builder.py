"""
Builds the metadata dict attached to every chunk before embedding/storage.

Schema matches the project spec exactly:
  chunk_id, document_id, bucket_id, file_name, file_type,
  page_number, content_type, source
"""

import hashlib
from pathlib import Path


def make_document_id(file_path: str) -> str:
    """Stable id derived from the file path, so re-ingesting the same
    document produces the same document_id (useful for idempotent
    re-runs / de-duping in the vector store).
    """
    digest = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:8]
    return f"doc_{digest}"


def make_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}_chunk_{index:04d}"


def build_metadata(chunk, index: int) -> dict:
    file_path = Path(chunk.document_path)
    document_id = make_document_id(chunk.document_path)

    metadata = {
        "chunk_id": make_chunk_id(document_id, index),
        "document_id": document_id,
        "bucket_id": chunk.bucket_id,
        "file_name": file_path.name,
        "file_type": file_path.suffix.lstrip(".").lower(),
        "content_type": chunk.content_type,
        "source": str(chunk.document_path),
    }

    # page_number is only meaningful for PDF-derived chunks.
    if chunk.page_number is not None:
        metadata["page_number"] = chunk.page_number

    return metadata
