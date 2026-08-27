"""
Splits cleaned RawContentUnit text into meaningful chunks for embedding.

Strategy:
- Tables remain complete and are never split.
- Text is split into sections when possible.
- Sections are split into sentences.
- Chunks try to preserve sentence boundaries.
- Overlap is sentence-based instead of character-based.
"""

import re
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


def chunk_units(
    units,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Convert preprocessed content units into embedding chunks.
    """

    chunks: List[Chunk] = []

    for unit in units:

        # -------------------------------------------------
        # TABLES
        # -------------------------------------------------
        # Tables must remain complete because splitting
        # rows/columns would destroy their relationships.
        if unit.content_type == "table":
            chunks.append(
                _to_chunk(unit, unit.text.strip())
            )
            continue

        # -------------------------------------------------
        # TEXT / OCR / IMAGE DESCRIPTION
        # -------------------------------------------------
        sections = _split_into_sections(unit.text)

        for section in sections:

            sentences = _split_into_sentences(section)

            section_chunks = _build_chunks_from_sentences(
                sentences=sentences,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for piece in section_chunks:

                if piece.strip():
                    chunks.append(
                        _to_chunk(unit, piece)
                    )

    return chunks


def _to_chunk(unit, text: str) -> Chunk:
    """Create a Chunk while preserving metadata."""

    return Chunk(
        text=text.strip(),
        document_path=str(unit.document_path),
        bucket_id=unit.bucket_id,
        content_type=unit.content_type,
        page_number=unit.page_number,
    )


# =========================================================
# SECTION SPLITTING
# =========================================================

def _split_into_sections(text: str) -> List[str]:
    """
    Split text into semantic sections.

    """

    if not text or not text.strip():
        return []

    text = text.strip()
    pattern = re.compile(
        r"""
        (?P<heading>
            [A-Z]
            [A-Za-z&/\-\s]{1,50}
        )
        \s+
        (?P<number>\d{1,2}\.)
        """,
        re.VERBOSE,
    )

    matches = list(pattern.finditer(text))

    # No headings found.
    if not matches:
        return [text]

    sections: List[str] = []

    # Preserve text before the first detected heading.
    first_start = matches[0].start()

    if text[:first_start].strip():
        sections.append(
            text[:first_start].strip()
        )

    # Create sections.
    for index, match in enumerate(matches):

        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        section = text[start:end].strip()

        if section:
            sections.append(section)

    return sections


# =========================================================
# SENTENCE SPLITTING
# =========================================================

def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Handles:
    - .
    - ?
    - !
    - Numbered facts such as '14.'
    """

    if not text or not text.strip():
        return []

    text = re.sub(
        r"\s+",
        " ",
        text.strip(),
    )

    # Split before numbered items:
    #
    # "... Geography 01. Earth is..."
    #
    # becomes:
    #
    # "Geography"
    # "01. Earth is..."
    text = re.sub(
        r"(?<!^)(?=\s\d{1,2}\.\s)",
        "\n",
        text,
    )

    # Split normal sentences.
    raw_sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    sentences: List[str] = []

    for sentence in raw_sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Avoid keeping tiny fragments separately.
        if (
            sentences
            and len(sentence) < 25
        ):
            sentences[-1] += " " + sentence
        else:
            sentences.append(sentence)

    return sentences


# =========================================================
# CHUNK BUILDING
# =========================================================

def _build_chunks_from_sentences(
    sentences: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Build chunks without cutting sentences.

    Overlap is based on complete trailing sentences,
    not arbitrary characters.
    """

    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        # ---------------------------------------------
        # SENTENCE TOO LARGE
        # ---------------------------------------------
        if sentence_length > chunk_size:

            if current:
                chunks.append(
                    " ".join(current)
                )
                current = []
                current_length = 0

            # Emergency fallback.
            # Only used when one individual sentence is
            # larger than chunk_size.
            chunks.extend(
                _split_long_sentence(
                    sentence,
                    chunk_size,
                )
            )

            continue

        # ---------------------------------------------
        # ADD TO CURRENT CHUNK
        # ---------------------------------------------
        separator_length = 1 if current else 0

        if (
            current_length
            + separator_length
            + sentence_length
            <= chunk_size
        ):
            current.append(sentence)
            current_length += (
                separator_length
                + sentence_length
            )

            continue

        # ---------------------------------------------
        # CHUNK FULL
        # ---------------------------------------------
        if current:

            chunks.append(
                " ".join(current)
            )

        # ---------------------------------------------
        # CREATE SENTENCE-BASED OVERLAP
        # ---------------------------------------------
        overlap_sentences = _get_overlap_sentences(
            current,
            chunk_overlap,
        )

        current = overlap_sentences + [sentence]

        current_length = len(
            " ".join(current)
        )

    # Add final chunk.
    if current:

        final_chunk = " ".join(current)

        if final_chunk.strip():

            chunks.append(final_chunk)

    return chunks


# =========================================================
# SENTENCE OVERLAP
# =========================================================

def _get_overlap_sentences(
    sentences: List[str],
    overlap_size: int,
) -> List[str]:
    """
    Keep complete sentences from the end of the previous
    chunk until approximately overlap_size characters
    are included.
    """

    if not sentences:
        return []

    overlap: List[str] = []
    total = 0

    for sentence in reversed(sentences):

        sentence_length = len(sentence)

        if (
            overlap
            and total + sentence_length > overlap_size
        ):
            break

        overlap.insert(0, sentence)

        total += sentence_length

    return overlap


# =========================================================
# FALLBACK FOR VERY LONG SENTENCES
# =========================================================

def _split_long_sentence(
    text: str,
    chunk_size: int,
) -> List[str]:
    """
    Emergency fallback for a single sentence that is
    longer than chunk_size.

    Splits at word boundaries where possible.
    """

    words = text.split()

    chunks: List[str] = []
    current: List[str] = []
    current_length = 0

    for word in words:

        word_length = len(word)

        separator_length = (
            1
            if current
            else 0
        )

        if (
            current
            and current_length
            + separator_length
            + word_length
            > chunk_size
        ):

            chunks.append(
                " ".join(current)
            )

            current = []
            current_length = 0

        current.append(word)

        current_length += (
            separator_length
            + word_length
        )

    if current:

        chunks.append(
            " ".join(current)
        )

    return chunks