import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from src.embeddings.embedder import embed_query
from src.vector_store import get_vector_store
from src.retrieval.retriever import _rerank


def main():

    if len(sys.argv) < 2:
        print(
            'Usage: python -m scripts.test_retrieval_debug "your question"'
        )
        sys.exit(1)

    query = sys.argv[1]

    print("=" * 70)
    print("RETRIEVAL DEBUG TEST")
    print("=" * 70)

    print(f"\nQuery: {query}")

    # Step 1: Create embedding
    print("\nCreating query embedding...")

    query_embedding = embed_query(query)

    # Step 2: Raw Chroma search
    print("\nSearching ChromaDB...")

    store = get_vector_store()

    candidates = store.query(
        query_embedding=query_embedding,
        top_k=10,
        bucket_id=None,
    )

    print("\n" + "=" * 70)
    print("RAW CHROMADB RESULTS")
    print("=" * 70)

    if not candidates:
        print("No candidates returned from ChromaDB.")
        return

    for i, candidate in enumerate(
        candidates,
        start=1,
    ):
        metadata = candidate.get("metadata", {})

        print(f"\nCANDIDATE {i}")
        print("-" * 70)
        print(
            "Distance:",
            candidate.get("distance")
        )
        print(
            "File:",
            metadata.get("file_name")
        )
        print(
            "Bucket:",
            metadata.get("bucket_id")
        )
        print(
            "Page:",
            metadata.get("page_number")
        )

        text = candidate.get(
            "text",
            ""
        )

        print("\nText:")
        print(text[:500])

    # Step 3: CrossEncoder reranking
    print("\n" + "=" * 70)
    print("CROSSENCODER RERANKED RESULTS")
    print("=" * 70)

    ranked = _rerank(
        query=query,
        candidates=candidates,
    )

    for i, result in enumerate(
        ranked,
        start=1,
    ):
        metadata = result.get(
            "metadata",
            {}
        )

        print(f"\nRANK {i}")
        print("-" * 70)
        print(
            "Relevance score:",
            result.get("relevance_score")
        )
        print(
            "Vector distance:",
            result.get("vector_distance")
        )
        print(
            "File:",
            metadata.get("file_name")
        )

        print("\nText:")
        print(
            result.get("text", "")[:500]
        )


if __name__ == "__main__":
    main()