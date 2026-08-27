from pathlib import Path

from src.ingestion.pipeline import ingest_document

from src.preprocessing.preprocessing_pipeline import (
    preprocess_content_units,
)

from src.chunking.chunker import chunk_units

# IMPORTANT:
# Use the actual import path where your existing
# embed_texts() function is located.
from src.embeddings.embedder import embed_texts


def main():

    print("=" * 70)
    print("REAL CHUNK → EMBEDDING TEST")
    print("=" * 70)

    # Use the same document you used for chunking.
    file_path = Path(
        "data/bucket_1/doc1.pdf"
    )

    bucket_id = "bucket_1"

    # --------------------------------------------------
    # 1. INGEST
    # --------------------------------------------------

    print("\n1. INGESTING DOCUMENT")
    print("-" * 70)

    raw_units = ingest_document(
        file_path=file_path,
        bucket_id=bucket_id,
    )

    print(f"Raw units: {len(raw_units)}")

    # --------------------------------------------------
    # 2. PREPROCESS
    # --------------------------------------------------

    print("\n2. PREPROCESSING")
    print("-" * 70)

    processed_units = preprocess_content_units(
        raw_units
    )

    print(f"Processed units: {len(processed_units)}")

    # --------------------------------------------------
    # 3. CHUNK
    # --------------------------------------------------

    print("\n3. CHUNKING")
    print("-" * 70)

    chunks = chunk_units(
        processed_units
    )

    print(f"Chunks created: {len(chunks)}")

    # --------------------------------------------------
    # 4. EXTRACT TEXT
    # --------------------------------------------------

    texts = [
        chunk.text
        for chunk in chunks
        if chunk.text.strip()
    ]

    print(f"Texts to embed: {len(texts)}")

    # --------------------------------------------------
    # 5. EMBED
    # --------------------------------------------------

    print("\n4. CREATING EMBEDDINGS")
    print("-" * 70)

    vectors = embed_texts(texts)

    print(f"\nVectors created: {len(vectors)}")

    # --------------------------------------------------
    # 6. VERIFY
    # --------------------------------------------------

    print("\n5. VERIFYING RESULTS")
    print("-" * 70)

    for index, (chunk, vector) in enumerate(
        zip(chunks, vectors),
        start=1,
    ):

        print(f"\nCHUNK {index}")
        print(f"Type: {chunk.content_type}")
        print(f"Page: {chunk.page_number}")
        print(f"Text length: {len(chunk.text)}")
        print(f"Vector dimension: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")

    print("\n" + "=" * 70)
    print("REAL CHUNK → EMBEDDING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()