import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking.chunker import chunk_units
from src.ingestion.pipeline import RawContentUnit


def _unit(text, content_type="text", page_number=1):
    return RawContentUnit(
        document_path=Path("data/bucket_1/sample.pdf"),
        bucket_id="bucket_1",
        content_type=content_type,
        text=text,
        page_number=page_number,
    )


def test_short_text_stays_one_chunk():
    unit = _unit("This is a short paragraph.")
    chunks = chunk_units([unit], chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "This is a short paragraph."


def test_long_text_gets_split():
    long_text = "Paragraph one.\n\n" + ("word " * 400) + "\n\nParagraph three."
    unit = _unit(long_text)
    chunks = chunk_units([unit], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 260  # allow slight overshoot from paragraph joins


def test_table_never_split():
    table_text = "| a | b |\n| --- | --- |\n| 1 | 2 |" * 50  # long, but a table
    unit = _unit(table_text, content_type="table")
    chunks = chunk_units([unit], chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0].content_type == "table"


def test_metadata_carried_through():
    unit = _unit("Some content.", page_number=7)
    chunks = chunk_units([unit])
    assert chunks[0].bucket_id == "bucket_1"
    assert chunks[0].page_number == 7
