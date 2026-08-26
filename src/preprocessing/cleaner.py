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
    """Remove standalone page markers."""
    lines = text.split("\n")

    cleaned = [
        line
        for line in lines
        if not PAGE_MARKER_PATTERN.match(line)
    ]

    return "\n".join(cleaned)


def remove_repeated_lines(text: str) -> str:
    """Remove immediately repeated non-empty lines."""
    lines = text.split("\n")
    cleaned = []
    previous_non_empty_line = None

    for line in lines:
        normalized = line.strip()

        if not normalized:
            cleaned.append(line)
            continue

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
    """Remove repeated headers appearing near the top of pages."""

    if not page_texts:
        return []

    candidate_counts: dict[str, int] = {}

    for page_text in page_texts:
        lines = page_text.split("\n")
        seen_on_page = set()

        for line in lines[:top_lines]:
            normalized = re.sub(r"\s+", " ", line.strip())

            if not normalized:
                continue

            if re.match(r"^\d+[**\.\)]\s*", normalized):
                continue

            if len(normalized) > 100:
                continue

            key = normalized.lower()

            if key not in seen_on_page:
                candidate_counts[key] = candidate_counts.get(key, 0) + 1
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
    """Apply safe preprocessing to a single page."""

    if not text:
        return ""

    text = remove_page_markers(text)
    text = normalize_whitespace(text)
    text = remove_repeated_lines(text)

    return text.strip()


def clean_pages(page_texts: list[str]) -> list[str]:
    """Clean document pages and remove repeated headers."""

    pages = remove_repeated_headers(page_texts)

    return [clean_text(page) for page in pages]