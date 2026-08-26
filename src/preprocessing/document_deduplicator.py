"""Document-level deduplication."""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List
import re


DUPLICATE_THRESHOLD = 0.90


@dataclass
class DeduplicatedBlock:
    """A document content block."""

    content: str
    page_number: int
    block_type: str  # "text" or "table"


def normalize_for_comparison(text: str) -> str:
    """Normalize text for duplicate comparison."""

    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


def similarity(a: str, b: str) -> float:
    """Return similarity between two strings."""

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def split_into_blocks(text: str) -> List[str]:
    """Split text into logical blocks."""

    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    blocks = []
    current = []

    def flush() -> None:
        if current:
            content = " ".join(current).strip()

            if content:
                blocks.append(content)

            current.clear()

    for line in lines:
        if not line:
            flush()
            continue

        current.append(line)

    flush()

    return blocks


def is_duplicate(
    content: str,
    existing_blocks: List[DeduplicatedBlock],
    threshold: float = DUPLICATE_THRESHOLD,
) -> bool:
    """Check whether content is a duplicate."""

    normalized = normalize_for_comparison(content)

    if not normalized:
        return False

    for existing in existing_blocks:
        if existing.block_type != "text":
            continue

        existing_normalized = normalize_for_comparison(
            existing.content
        )

        if not existing_normalized:
            continue

        if normalized == existing_normalized:
            return True

        if (
            len(normalized) < 60
            or len(existing_normalized) < 60
        ):
            continue

        score = similarity(
            normalized,
            existing_normalized,
        )

        if score >= threshold:
            return True

    return False


def deduplicate_blocks(
    blocks: List[DeduplicatedBlock],
    threshold: float = DUPLICATE_THRESHOLD,
) -> List[DeduplicatedBlock]:
    """Remove duplicate text blocks."""

    kept = []

    for block in blocks:
        if not block.content.strip():
            continue

        if block.block_type == "table":
            kept.append(block)
            continue

        if is_duplicate(
            block.content,
            kept,
            threshold,
        ):
            continue

        kept.append(block)

    return kept


def deduplicate_document_text(
    page_number: int,
    text: str,
    existing_blocks: List[DeduplicatedBlock],
    threshold: float = DUPLICATE_THRESHOLD,
) -> List[DeduplicatedBlock]:
    """Split page text and remove duplicates."""

    page_blocks = split_into_blocks(text)
    new_blocks = []

    for content in page_blocks:
        if is_duplicate(
            content,
            existing_blocks + new_blocks,
            threshold,
        ):
            continue

        new_blocks.append(
            DeduplicatedBlock(
                content=content,
                page_number=page_number,
                block_type="text",
            )
        )

    return new_blocks