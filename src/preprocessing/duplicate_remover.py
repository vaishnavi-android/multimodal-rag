from typing import List
import re


def _normalize_for_comparison(text: str) -> str:
    """Create a comparison form without changing the original text."""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def remove_duplicate_blocks(
    blocks: List[str],
    similarity_threshold: float = 0.90,
) -> List[str]:
    """
    Remove highly similar duplicate text blocks.

    The original text is preserved for blocks that survive.

    This is intentionally conservative:
    - exact duplicates are removed
    - highly similar blocks are removed
    - unique content is retained
    """

    if not blocks:
        return []

    cleaned = []
    normalized_seen = []

    for block in blocks:
        if not block or not block.strip():
            continue

        normalized = _normalize_for_comparison(block)

        if not normalized:
            continue

        # Exact duplicate
        if normalized in normalized_seen:
            continue

        # Conservative similarity check
        is_duplicate = False

        for previous in normalized_seen:
            similarity = _similarity(normalized, previous)

            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            cleaned.append(block)
            normalized_seen.append(normalized)

    return cleaned


def _similarity(a: str, b: str) -> float:
    """
    Calculate a simple character-based similarity.

    Uses SequenceMatcher from Python's standard library,
    so no additional dependency is required.
    """

    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()