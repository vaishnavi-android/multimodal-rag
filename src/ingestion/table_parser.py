"""Table extraction from PDFs."""

from pathlib import Path
from typing import List


def extract_tables(file_path: Path, page_number: int) -> List[str]:
    """
    Extract valid tables from a PDF page as Markdown.

    Complex page layouts can sometimes be incorrectly detected as tables,
    so each extracted table is validated before being returned.
    """

    import pdfplumber

    tables_md = []

    with pdfplumber.open(Path(file_path)) as pdf:
        page = pdf.pages[page_number - 1]

        tables = page.extract_tables()

        for table in tables:

            # Skip empty tables
            if not _is_valid_table(table):
                continue

            markdown = _table_to_markdown(table)

            if markdown:
                tables_md.append(markdown)

    return tables_md


def _is_valid_table(table: List[List[str]]) -> bool:
    """
    Check whether extracted data looks like a meaningful table.

    Rejects false tables caused by:
    - textbook/page layouts
    - multi-column content
    - diagrams
    - decorative elements
    - paragraphs incorrectly grouped into cells
    """

    if not table:
        return False

    # --------------------------------------------------
    # 1. Clean empty rows
    # --------------------------------------------------

    rows = []

    for row in table:
        if not row:
            continue

        cleaned_row = []

        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                cleaned_row.append(str(cell).strip())

        if any(cleaned_row):
            rows.append(cleaned_row)

    if len(rows) < 2:
        return False

    # --------------------------------------------------
    # 2. Check column structure
    # --------------------------------------------------

    max_columns = max(len(row) for row in rows)

    if max_columns < 2:
        return False

    # Normalize rows temporarily
    normalized_rows = []

    for row in rows:
        normalized_row = row + [""] * (
            max_columns - len(row)
        )
        normalized_rows.append(normalized_row)

    # --------------------------------------------------
    # 3. Basic cell statistics
    # --------------------------------------------------

    total_cells = 0
    non_empty_cells = 0
    cell_lengths = []

    for row in normalized_rows:
        for cell in row:
            total_cells += 1

            if cell.strip():
                non_empty_cells += 1
                cell_lengths.append(len(cell))

    if non_empty_cells == 0:
        return False

    fill_ratio = non_empty_cells / total_cells

    # Too sparse = likely layout fragments
    if fill_ratio < 0.30:
        return False

    # --------------------------------------------------
    # 4. Reject paragraph-heavy structures
    # --------------------------------------------------

    average_cell_length = (
        sum(cell_lengths) / len(cell_lengths)
    )

    if average_cell_length > 300:
        return False

    # Find largest cell
    largest_cell = max(cell_lengths)

    # A very large cell combined with a sparse table
    # often means page text/layout was detected as a table
    if largest_cell > 800 and fill_ratio < 0.70:
        return False

    # --------------------------------------------------
    # 5. Detect suspicious row structure
    # --------------------------------------------------

    # Count rows where only one cell contains most content.
    # This is common in page layouts falsely detected as tables.
    single_content_rows = 0

    for row in normalized_rows:

        filled_cells = [
            cell for cell in row
            if cell.strip()
        ]

        if len(filled_cells) == 1:
            single_content_rows += 1

    single_content_ratio = (
        single_content_rows / len(normalized_rows)
    )

    if single_content_ratio > 0.70:
        return False

    # --------------------------------------------------
    # 6. Detect suspicious extremely uneven cells
    # --------------------------------------------------

    # Example fake table:
    #
    # | huge paragraph | | |
    # | short text     | | |
    #
    # Real tables generally have more balanced cell structure.

    if len(cell_lengths) >= 2:

        smallest_cell = min(cell_lengths)

        # Avoid division by zero
        if smallest_cell > 0:

            size_ratio = largest_cell / smallest_cell

            # Extremely uneven content distribution
            # can indicate a page layout.
            if (
                size_ratio > 80
                and largest_cell > 300
            ):
                return False

    # --------------------------------------------------
    # VALID TABLE
    # --------------------------------------------------

    return True


def _table_to_markdown(table: List[List[str]]) -> str:
    """Convert a table into Markdown format."""

    rows = [
        [
            str(cell).replace("\n", " ").strip()
            if cell is not None
            else ""
            for cell in row
        ]
        for row in table
    ]

    # Remove empty rows
    rows = [
        row for row in rows
        if any(cell.strip() for cell in row)
    ]

    if not rows:
        return ""

    # Make all rows the same length
    max_columns = max(len(row) for row in rows)

    normalized_rows = []

    for row in rows:

        row = row + [""] * (max_columns - len(row))

        normalized_rows.append(row)

    header, *body = normalized_rows

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * max_columns) + " |",
    ]

    for row in body:
        lines.append(
            "| " + " | ".join(row) + " |"
        )

    return "\n".join(lines)