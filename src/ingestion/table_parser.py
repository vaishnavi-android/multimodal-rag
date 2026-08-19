"""
Table extraction from PDFs. Kept separate from pdf_parser.py because
table structure needs different handling downstream (e.g. rendered as
markdown so row/column relationships survive chunking, per the spec's
"preserve table relationships" requirement).
"""

from pathlib import Path
from typing import List


def extract_tables(file_path: Path, page_number: int) -> List[str]:
    """Extract tables from a specific PDF page, each returned as a
    markdown-formatted string so relationships between cells are
    preserved through chunking and embedding.

    Requires: pip install pdfplumber
    """
    import pdfplumber

    tables_md = []
    with pdfplumber.open(Path(file_path)) as pdf:
        page = pdf.pages[page_number - 1]
        tables = page.extract_tables()

        for table in tables:
            if not table or not any(any(cell for cell in row) for row in table):
                continue
            tables_md.append(_table_to_markdown(table))

    return tables_md


def _table_to_markdown(table: List[List[str]]) -> str:
    rows = [[cell if cell is not None else "" for cell in row] for row in table]
    if not rows:
        return ""

    header, *body = rows
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
