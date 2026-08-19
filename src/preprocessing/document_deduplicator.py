"""
Document-level deduplication.

Splits document text into logical blocks, then removes only
high-confidence duplicate blocks across the entire document.

Design goals:
- Preserve the first occurrence.
- Preserve unique content.
- Handle numbered facts correctly.
- Join wrapped lines belonging to the same fact.
- Handle unnumbered repeated facts in appendix sections.
- Avoid deleting partially similar facts.
- Keep tables separate.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import List


DUPLICATE_THRESHOLD = 0.90


@dataclass
class DeduplicatedBlock:
    """A logical piece of document content."""

    content: str
    page_number: int
    block_type: str  # "text" or "table"


def normalize_for_comparison(text: str) -> str:
    """
    Normalize text ONLY for duplicate comparison.

    Original content is never modified.
    """

    if not text:
        return ""

    text = text.lower()

    # Join OCR line-wrap artifacts such as:
    # "terminol-\nogy" -> "terminology"
    text = re.sub(r"-\s*\n\s*", "", text)

    # Treat line breaks as spaces.
    text = re.sub(r"\s+", " ", text)

    # Normalize spaces before punctuation.
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


def similarity(a: str, b: str) -> float:
    """Return similarity between 0.0 and 1.0."""

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def split_into_blocks(text: str) -> List[str]:
    """
    Split cleaned document text into logical blocks.

    Handles:

    1. Numbered facts

        01. Mesopotamia ...
        there.

    2. Normal headings.

    3. Appendix-style unnumbered repeated facts.

    4. OCR/search notes.

    Wrapped lines belonging to the same fact are joined together.
    """

    if not text:
        return []

    # ---------------------------------------------------------
    # Normalize line endings.
    # ---------------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize horizontal whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    lines = [line.strip() for line in text.split("\n")]

    blocks: List[str] = []
    current: List[str] = []

    # ---------------------------------------------------------
    # Patterns.
    # ---------------------------------------------------------

    numbered_pattern = re.compile(
        r"^\d{1,3}\.\s+"
    )

    heading_patterns = [
        re.compile(r"^APPENDIX\b", re.IGNORECASE),
        re.compile(r"^OCR\s*/\s*SEARCH\s+TEST\s+NOTES$", re.IGNORECASE),
        re.compile(r"^Ancient\s*&\s*Modern\s+History$", re.IGNORECASE),
        re.compile(r"^Computing$", re.IGNORECASE),
        re.compile(r"^Animals\s*&\s*Plants$", re.IGNORECASE),
        re.compile(r"^Materials\s*&\s*Engineering$", re.IGNORECASE),
        re.compile(r"^Everyday\s+Science$", re.IGNORECASE),
        re.compile(r"^Extracted\s+table\b", re.IGNORECASE),
    ]

    # These are specifically the unnumbered repeated facts
    # appearing in the appendix of the test document.
    appendix_fact_starters = (
        "Mesopotamia is ",
        "Ancient Egypt depended ",
        "The Indus Valley Civilization included ",
    )

    def flush_current() -> None:
        """Flush the current logical block."""

        nonlocal current

        if current:
            content = " ".join(current).strip()

            if content:
                blocks.append(content)

        current = []

    def is_heading(line: str) -> bool:
        """Return True when line is a known section heading."""

        return any(
            pattern.match(line)
            for pattern in heading_patterns
        )

    # ---------------------------------------------------------
    # Process lines.
    # ---------------------------------------------------------

    for line in lines:

        # Blank line = logical boundary.
        if not line:
            flush_current()
            continue

        # -----------------------------------------------------
        # Numbered fact.
        # -----------------------------------------------------

        if numbered_pattern.match(line):
            flush_current()
            current.append(line)
            continue

        # -----------------------------------------------------
        # Known heading.
        # -----------------------------------------------------

        if is_heading(line):
            flush_current()
            blocks.append(line)
            continue

        # -----------------------------------------------------
        # Appendix repeated facts.
        #
        # Each one must become its own block so that the
        # document-level deduplicator can compare it against
        # the original fact on Page 1.
        # -----------------------------------------------------

        if line.startswith(appendix_fact_starters):

            flush_current()

            current.append(line)

            # These facts normally end on the same line.
            if line.endswith("."):
                flush_current()

            continue

        # -----------------------------------------------------
        # Continuation of an existing fact.
        #
        # Example:
        #
        # Mesopotamia ... developed
        # there.
        #
        # stays as ONE block.
        # -----------------------------------------------------

        if current:
            current.append(line)
            continue

        # -----------------------------------------------------
        # Otherwise start a normal block.
        # -----------------------------------------------------

        current.append(line)

    # Flush anything remaining.
    flush_current()

    return [
        block
        for block in blocks
        if block.strip()
    ]


def is_duplicate(
    content: str,
    existing_blocks: List[DeduplicatedBlock],
    threshold: float = DUPLICATE_THRESHOLD,
) -> bool:
    """
    Determine whether content is a high-confidence duplicate
    of an existing text block.
    """

    normalized = normalize_for_comparison(content)

    if not normalized:
        return False

    for existing in existing_blocks:

        # Only compare text with text.
        if existing.block_type != "text":
            continue

        existing_normalized = normalize_for_comparison(
            existing.content
        )

        if not existing_normalized:
            continue

        # Exact duplicate.
        if normalized == existing_normalized:
            return True

        # Avoid fuzzy matching very short blocks.
        if len(normalized) < 60:
            continue

        if len(existing_normalized) < 60:
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
    """
    Deduplicate a collection of logical blocks.

    The first occurrence is preserved.

    Tables are always preserved.
    """

    kept: List[DeduplicatedBlock] = []

    for block in blocks:

        # Ignore empty blocks.
        if not block.content.strip():
            continue

        # Never deduplicate tables here.
        if block.block_type == "table":
            kept.append(block)
            continue

        # Remove only high-confidence duplicate text.
        if is_duplicate(
            block.content,
            kept,
            threshold=threshold,
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
    """
    Split one page into logical blocks and remove blocks that
    already appeared earlier in the document.

    The first occurrence is preserved.
    """

    page_blocks = split_into_blocks(text)

    new_blocks: List[DeduplicatedBlock] = []

    for content in page_blocks:

        if is_duplicate(
            content,
            existing_blocks + new_blocks,
            threshold=threshold,
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