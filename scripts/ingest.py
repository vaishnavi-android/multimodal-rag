"""
Runs the full ingestion pipeline for one or all buckets.

Pipeline:

    documents
        ↓
    ingestion
        ↓
    multimodal preprocessing
        ↓
    chunking
        ↓
    metadata
        ↓
    embeddings
        ↓
    vector database

Each document is processed independently so that
document-level deduplication never affects another document.

Usage:

    python scripts/ingest.py

    python scripts/ingest.py --bucket bucket_1
"""

import argparse
import sys
from pathlib import Path


# Allow running:
#     python scripts/ingest.py
# from the project root.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)


from src.config.settings import BUCKETS
from src.ingestion.file_detector import list_bucket_documents
from src.ingestion.pipeline import ingest_document
from src.preprocessing.preprocessing_pipeline import (
    preprocess_content_units,
)
from src.chunking.chunker import chunk_units
from src.metadata.metadata_builder import build_metadata
from src.embeddings.embedder import embed_texts
from src.vector_store import get_vector_store


def run_ingestion(bucket_id: str):

    if bucket_id not in BUCKETS:
        raise ValueError(
            f"Unknown bucket_id '{bucket_id}'. "
            f"Valid: {list(BUCKETS)}"
        )

    bucket_dir = BUCKETS[bucket_id]

    documents = list_bucket_documents(bucket_dir)

    print(
        f"[{bucket_id}] "
        f"Found {len(documents)} document(s) "
        f"in {bucket_dir}"
    )

    store = get_vector_store()

    total_chunks = 0

    # =========================================================
    # PROCESS EACH DOCUMENT INDEPENDENTLY
    # =========================================================

    for doc_path in documents:

        print("\n" + "=" * 70)
        print(
            f"[{bucket_id}] PROCESSING DOCUMENT: "
            f"{doc_path.name}"
        )
        print("=" * 70)

        # =====================================================
        # 1. INGEST RAW CONTENT
        # =====================================================

        print(
            f"[{bucket_id}] "
            f"Parsing {doc_path.name} ..."
        )

        units = ingest_document(
            doc_path,
            bucket_id,
        )

        print(
            f"[{bucket_id}] "
            f"{len(units)} raw content unit(s) extracted"
        )

        # =====================================================
        # 2. MULTIMODAL PREPROCESSING
        # =====================================================

        units = preprocess_content_units(
            units
        )

        print(
            f"[{bucket_id}] "
            f"{len(units)} content unit(s) "
            f"after preprocessing"
        )

        # =====================================================
        # 3. CHUNKING
        # =====================================================

        chunks = chunk_units(units)

        print(
            f"[{bucket_id}] "
            f"{len(chunks)} chunk(s) created"
        )

        if not chunks:
            print(
                f"[{bucket_id}] "
                f"No chunks generated for "
                f"{doc_path.name}. Skipping."
            )
            continue

        # =====================================================
        # 4. METADATA
        # =====================================================

        texts = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            build_metadata(chunk, i)
            for i, chunk in enumerate(chunks)
        ]

        ids = [
            metadata["chunk_id"]
            for metadata in metadatas
        ]

        # =====================================================
        # 5. EMBEDDINGS
        # =====================================================

        print(
            f"[{bucket_id}] "
            f"Generating embeddings for "
            f"{doc_path.name} ..."
        )

        embeddings = embed_texts(texts)

        # =====================================================
        # 6. VECTOR STORE
        # =====================================================

        print(
            f"[{bucket_id}] "
            f"Writing {len(chunks)} chunk(s) "
            f"to vector store ..."
        )

        store.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)

        print(
            f"[{bucket_id}] "
            f"Finished {doc_path.name}"
        )

    # =========================================================
    # BUCKET SUMMARY
    # =========================================================

    print("\n" + "=" * 70)
    print(
        f"[{bucket_id}] INGESTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Documents processed: {len(documents)}"
    )

    print(
        f"Chunks created this run: {total_chunks}"
    )

    print(
        f"Total chunks now in store: "
        f"{store.count(bucket_id=bucket_id)}"
    )

    return len(documents), total_chunks


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Ingest documents into the vector store."
        )
    )

    parser.add_argument(
        "--bucket",
        choices=list(BUCKETS.keys()),
        default=None,
        help=(
            "Ingest a single bucket. "
            "Omit to ingest all buckets."
        ),
    )

    args = parser.parse_args()

    targets = (
        [args.bucket]
        if args.bucket
        else list(BUCKETS.keys())
    )

    for bucket_id in targets:
        run_ingestion(bucket_id)