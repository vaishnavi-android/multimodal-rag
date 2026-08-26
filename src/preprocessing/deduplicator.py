"""
Generic document-level text deduplication.

Removes only high-confidence duplicate text while preserving
the first occurrence.

The implementation is intentionally generic and does not contain
document-specific headings, facts, or topic rules.
"""

import re
from difflib import SequenceMatcher
from typing import List


DUPLICATE_THRESHOLD = 0.92


def normalize_for_comparison(text: str) -> str:
    """
    Create a normalized version of text for duplicate comparison.

    The original text is never modified.
    """

    if not text:
        return ""

    text = text.lower()

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    # Normalize spaces before punctuation.
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


def is_duplicate(
    text: str,
    existing_texts: List[str],
    threshold: float = DUPLICATE_THRESHOLD,
) -> bool:
    """
    Check whether text is a high-confidence duplicate of
    previously accepted text.
    """

    normalized = normalize_for_comparison(text)

    if not normalized:
        return False

    for existing in existing_texts:

        existing_normalized = normalize_for_comparison(existing)

        if not existing_normalized:
            continue

        # Exact duplicate.
        if normalized == existing_normalized:
            return True

        # Avoid fuzzy matching very short text.
        if len(normalized) < 60:
            continue

        if len(existing_normalized) < 60:
            continue

        similarity = SequenceMatcher(
            None,
            normalized,
            existing_normalized,
        ).ratio()

        if similarity >= threshold:
            return True

    return False


def split_text_blocks(text: str) -> List[str]:
    """
    Split text into generic logical blocks.

    Paragraph boundaries are preferred. If no paragraph boundaries
    exist, the complete text remains one block.
    """

    if not text or not text.strip():
        return []

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Split on blank lines.
    blocks = re.split(r"\n\s*\n+", text)

    cleaned_blocks = []

    for block in blocks:
        block = re.sub(r"[ \t]+", " ", block)
        block = re.sub(r"\n+", " ", block)
        block = block.strip()

        if block:
            cleaned_blocks.append(block)

    return cleaned_blocks


def deduplicate_document_text(
    text: str,
    existing_texts: List[str],
) -> List[str]:
    """
    Split text into generic blocks and remove blocks that already
    appeared earlier in the document.

    Returns only newly accepted unique blocks.
    """

    new_blocks = []

    for block in split_text_blocks(text):

        comparison_pool = existing_texts + new_blocks

        if is_duplicate(block, comparison_pool):
            continue

        new_blocks.append(block)

    return new_blocks