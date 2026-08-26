"""Table extraction from PDFs."""

from pathlib import Path
from typing import List


def extract_tables(file_path: Path, page_number: int) -> List[str]:
    """Extract tables from a PDF page as Markdown."""

    import pdfplumber

    tables_md = []

    with pdfplumber.open(Path(file_path)) as pdf:
        page = pdf.pages[page_number - 1]
        tables = page.extract_tables()

        for table in tables:
            if not table or not any(
                any(cell for cell in row)
                for row in table
            ):
                continue

            tables_md.append(_table_to_markdown(table))

    return tables_md


def _table_to_markdown(table: List[List[str]]) -> str:
    """Convert a table into Markdown format."""

    rows = [
        [cell if cell is not None else "" for cell in row]
        for row in table
    ]

    if not rows:
        return ""

    header, *body = rows

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)