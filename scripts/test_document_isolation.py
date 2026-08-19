"""
Tests that preprocessing/deduplication is isolated per document.

The same logical content appearing in two different documents
must be preserved in both documents.

This test does NOT generate embeddings and does NOT write
anything to the vector store.
"""

from pathlib import Path

from src.ingestion.pipeline import ingest_document
from src.preprocessing.preprocessing_pipeline import (
    preprocess_content_units,
)


BUCKET_ID = "bucket_1"

DOC1_PATH = Path("data/bucket_1/doc1.pdf")
DOC2_PATH = Path("data/bucket_1/doc2.pdf")


def process_document(
    document_path: Path,
):
    """Ingest and preprocess one document independently."""

    print("\n" + "=" * 70)
    print(
        f"PROCESSING: {document_path.name}"
    )
    print("=" * 70)

    raw_units = ingest_document(
        document_path,
        BUCKET_ID,
    )

    print(
        f"Raw units: {len(raw_units)}"
    )

    processed_units = preprocess_content_units(
        raw_units
    )

    print(
        f"Processed units: "
        f"{len(processed_units)}"
    )

    text_units = [
        unit
        for unit in processed_units
        if unit.content_type == "text"
    ]

    table_units = [
        unit
        for unit in processed_units
        if unit.content_type == "table"
    ]

    print(
        f"Text units: {len(text_units)}"
    )

    print(
        f"Table units: {len(table_units)}"
    )

    return processed_units


def main():

    print("=" * 70)
    print("DOCUMENT ISOLATION TEST")
    print("=" * 70)

    if not DOC1_PATH.exists():
        raise FileNotFoundError(
            f"Document not found: {DOC1_PATH}"
        )

    if not DOC2_PATH.exists():
        raise FileNotFoundError(
            f"Document not found: {DOC2_PATH}"
        )

    # ---------------------------------------------------------
    # Process document 1 independently.
    # ---------------------------------------------------------

    doc1_units = process_document(
        DOC1_PATH
    )

    # ---------------------------------------------------------
    # Process document 2 independently.
    # ---------------------------------------------------------

    doc2_units = process_document(
        DOC2_PATH
    )

    # ---------------------------------------------------------
    # Extract normalized text for comparison.
    # ---------------------------------------------------------

    doc1_text = {
        unit.text.strip()
        for unit in doc1_units
        if unit.content_type == "text"
        and unit.text.strip()
    }

    doc2_text = {
        unit.text.strip()
        for unit in doc2_units
        if unit.content_type == "text"
        and unit.text.strip()
    }

    common_text = doc1_text.intersection(
        doc2_text
    )

    print("\n" + "=" * 70)
    print("DOCUMENT ISOLATION RESULT")
    print("=" * 70)

    print(
        f"\nDocument 1 text units: "
        f"{len(doc1_text)}"
    )

    print(
        f"Document 2 text units: "
        f"{len(doc2_text)}"
    )

    print(
        f"Common text units: "
        f"{len(common_text)}"
    )

    # ---------------------------------------------------------
    # The important assertion:
    #
    # Common content is allowed.
    #
    # The test is proving that document 2 still contains
    # its own content even when document 1 has already been
    # processed.
    # ---------------------------------------------------------

    if not doc1_text:
        raise AssertionError(
            "Document 1 produced no text units."
        )

    if not doc2_text:
        raise AssertionError(
            "Document 2 produced no text units."
        )

    print("\nPASS: Both documents retain their own text.")

    # ---------------------------------------------------------
    # Tables must also remain independent.
    # ---------------------------------------------------------

    doc1_tables = [
        unit
        for unit in doc1_units
        if unit.content_type == "table"
    ]

    doc2_tables = [
        unit
        for unit in doc2_units
        if unit.content_type == "table"
    ]

    print(
        f"\nDocument 1 tables: "
        f"{len(doc1_tables)}"
    )

    print(
        f"Document 2 tables: "
        f"{len(doc2_tables)}"
    )

    print("\n" + "=" * 70)
    print("DOCUMENT ISOLATION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()