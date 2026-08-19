"""
Splits cleaned RawContentUnit text into chunks for embedding.

Uses configurable chunk_size / chunk_overlap (from settings.py, never
hard-coded here). Splits on paragraph boundaries first, falling back to
a sliding window only when a single paragraph exceeds chunk_size - this
respects the spec's "prefer natural boundaries" guidance without pulling
in a heavier dependency for Phase 1.

Table units (content_type == "table") are never split - a table is
kept as one coherent chunk regardless of size, since splitting a table
mid-row would destroy the row/column relationships the spec explicitly
says to preserve.
"""

from dataclasses import dataclass
from typing import List

from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    document_path: str
    bucket_id: str
    content_type: str
    page_number: int | None


def chunk_units(units, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []

    for unit in units:
        if unit.content_type == "table":
            chunks.append(_to_chunk(unit, unit.text))
            continue

        for piece in _split_text(unit.text, chunk_size, chunk_overlap):
            chunks.append(_to_chunk(unit, piece))

    return chunks


def _to_chunk(unit, text: str) -> Chunk:
    return Chunk(
        text=text,
        document_path=str(unit.document_path),
        bucket_id=unit.bucket_id,
        content_type=unit.content_type,
        page_number=unit.page_number,
    )


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    pieces: List[str] = []
    current = ""

    for para in paragraphs:
        # A single paragraph longer than chunk_size gets its own sliding-window split.
        if len(para) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_sliding_window(para, chunk_size, chunk_overlap))
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            pieces.append(current)
            current = para

    if current:
        pieces.append(current)

    return pieces


def _sliding_window(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    step = max(chunk_size - chunk_overlap, 1)
    return [text[i:i + chunk_size] for i in range(0, len(text), step) if text[i:i + chunk_size].strip()]
