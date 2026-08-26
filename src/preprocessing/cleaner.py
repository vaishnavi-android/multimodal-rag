import re


PAGE_MARKER_PATTERN = re.compile(
    r"^\s*Page\s+\d+\s*$",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line.replace("\t", " ") for line in lines]
    lines = [re.sub(r"[ ]{2,}", " ", line) for line in lines]

    cleaned_lines = []
    previous_blank = False

    for line in lines:
        is_blank = not line.strip()

        if is_blank:
            if previous_blank:
                continue

            previous_blank = True
            cleaned_lines.append("")
        else:
            previous_blank = False
            cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines).strip()


def remove_page_markers(text: str) -> str:
    """
    Remove standalone page markers such as:

        Page 1
        Page 2
        PAGE 3

    Page numbers are preserved separately in metadata.
    """

    lines = text.split("\n")

    cleaned = []

    for line in lines:
        if PAGE_MARKER_PATTERN.match(line):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def remove_repeated_lines(text: str) -> str:
    """
    Remove immediately repeated identical non-empty lines.

    Blank lines are preserved as separators and reset duplicate tracking.
    """

    lines = text.split("\n")

    cleaned = []
    previous_non_empty_line = None

    for line in lines:
        normalized = line.strip()

        # Preserve blank lines.
        # Reset duplicate tracking because the next occurrence
        # is no longer immediately repeated.
        if not normalized:
            cleaned.append(line)
            previous_non_empty_line = None
            continue

        # Remove only immediately repeated non-empty lines.
        if normalized == previous_non_empty_line:
            continue

        cleaned.append(line)
        previous_non_empty_line = normalized

    return "\n".join(cleaned)


def remove_repeated_headers(
    page_texts: list[str],
    minimum_occurrences: int = 2,
    top_lines: int = 5,
) -> list[str]:
    """
    Remove likely repeated page headers conservatively.

    A line is considered a repeated header only when:
    - it appears near the top of multiple pages
    - it appears on at least minimum_occurrences pages
    - it is relatively short
    - it is not a numbered content line

    This avoids deleting legitimate document content.
    """

    if not page_texts:
        return []

    candidate_counts: dict[str, int] = {}

    for page_text in page_texts:
        lines = page_text.split("\n")

        # Only inspect the first few lines of each page.
        top = lines[:top_lines]

        seen_on_page = set()

        for line in top:
            normalized = re.sub(r"\s+", " ", line.strip())

            if not normalized:
                continue

            # Don't treat numbered facts as headers.
            if re.match(r"^\d+[\.\)]\s*", normalized):
                continue

            # Don't treat long sentences as headers.
            if len(normalized) > 100:
                continue

            key = normalized.lower()

            if key not in seen_on_page:
                candidate_counts[key] = (
                    candidate_counts.get(key, 0) + 1
                )
                seen_on_page.add(key)

    repeated_headers = {
        key
        for key, count in candidate_counts.items()
        if count >= minimum_occurrences
    }

    cleaned_pages = []

    for page_text in page_texts:
        lines = page_text.split("\n")
        cleaned_lines = []

        for index, line in enumerate(lines):
            normalized = re.sub(r"\s+", " ", line.strip())
            key = normalized.lower()

            # Only remove a repeated header when it is actually
            # located near the top of the page.
            if (
                index < top_lines
                and normalized
                and key in repeated_headers
            ):
                continue

            cleaned_lines.append(line)

        cleaned_pages.append("\n".join(cleaned_lines))

    return cleaned_pages


def clean_text(text: str) -> str:
    """
    Apply safe single-content-unit preprocessing.

    Performs structural cleaning only.
    Does not perform aggressive spelling or OCR correction.
    """

    if not text:
        return ""

    text = remove_page_markers(text)
    text = normalize_whitespace(text)
    text = remove_repeated_lines(text)

    return text.strip()


def clean_pages(page_texts: list[str]) -> list[str]:
    """
    Clean multiple pages from the same document.

    Repeated headers are detected across pages first,
    then each page receives normal single-page cleaning.
    """

    pages = remove_repeated_headers(page_texts)

    return [clean_text(page) for page in pages]