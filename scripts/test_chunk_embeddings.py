"""
Tests the full pipeline from chunks to embeddings.
"""

from src.embeddings.embedder import embed_texts

# Adjust this import if your chunker file has a different name.
from src.chunking.chunker import chunk_units


def main():

    print("=" * 70)
    print("CHUNK → EMBEDDING TEST")
    print("=" * 70)

    # Temporary test chunks.
    # We are first verifying that the embedding module
    # works correctly with chunk text.
    texts = [
        "The heart has four chambers.",
        "Earth is the third planet from the Sun.",
        "Jupiter is the largest planet.",
    ]

    print("\nEmbedding texts...")

    vectors = embed_texts(texts)

    print(f"\nTexts embedded: {len(texts)}")
    print(f"Vectors created: {len(vectors)}")

    for index, vector in enumerate(vectors, start=1):

        print(f"\nVECTOR {index}")
        print(f"Dimension: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")

    print("\n" + "=" * 70)
    print("CHUNK → EMBEDDING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
