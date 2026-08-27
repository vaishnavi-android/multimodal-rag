from src.vector_store import get_vector_store
from src.embeddings.embedder import embed_texts


def main():
    print("=" * 70)
    print("CHROMADB TEST")
    print("=" * 70)

    # --------------------------------------------------
    # 1. CREATE VECTOR STORE
    # --------------------------------------------------

    print("\n1. INITIALIZING VECTOR STORE")
    print("-" * 70)

    store = get_vector_store()

    print("Vector store initialized successfully.")

    # --------------------------------------------------
    # 2. TEST DATA
    # --------------------------------------------------

    texts = [
        "Earth is the third planet from the Sun.",
        "The human heart has four chambers.",
        "Jupiter is the largest planet in the Solar System.",
    ]

    ids = [
        "test_1",
        "test_2",
        "test_3",
    ]

    metadatas = [
        {
            "document_id": "test_document",
            "document_path": "test.pdf",
            "bucket_id": "bucket_1",
            "content_type": "text",
            "page_number": 1,
        },
        {
            "document_id": "test_document",
            "document_path": "test.pdf",
            "bucket_id": "bucket_1",
            "content_type": "text",
            "page_number": 1,
        },
        {
            "document_id": "test_document",
            "document_path": "test.pdf",
            "bucket_id": "bucket_2",
            "content_type": "text",
            "page_number": 2,
        },
    ]

    # --------------------------------------------------
    # 3. CREATE EMBEDDINGS
    # --------------------------------------------------

    print("\n2. CREATING EMBEDDINGS")
    print("-" * 70)

    embeddings = embed_texts(texts)

    print(f"Embeddings created: {len(embeddings)}")
    print(f"Vector dimension: {len(embeddings[0])}")

    # --------------------------------------------------
    # 4. STORE IN CHROMADB
    # --------------------------------------------------

    print("\n3. STORING DATA")
    print("-" * 70)

    store.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print("Chunks stored successfully.")

    # --------------------------------------------------
    # 5. CHECK COUNT
    # --------------------------------------------------

    print("\n4. CHECKING DATABASE")
    print("-" * 70)

    print(f"Total chunks: {store.count()}")
    print(f"Bucket 1 chunks: {store.count('bucket_1')}")
    print(f"Bucket 2 chunks: {store.count('bucket_2')}")

    # --------------------------------------------------
    # 6. QUERY TEST
    # --------------------------------------------------

    print("\n5. TESTING RETRIEVAL")
    print("-" * 70)

    query = "Which planet is the biggest?"

    query_embedding = embed_texts([query])[0]

    results = store.query(
        query_embedding=query_embedding,
        top_k=3,
    )

    print(f"\nQuery: {query}")

    print("\nResults:")

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Text: {result['text']}")
        print(f"Bucket: {result['metadata']['bucket_id']}")
        print(f"Distance: {result['distance']}")

    # --------------------------------------------------
    # 7. BUCKET FILTER TEST
    # --------------------------------------------------

    print("\n6. TESTING BUCKET FILTER")
    print("-" * 70)

    results = store.query(
        query_embedding=query_embedding,
        top_k=3,
        bucket_id="bucket_1",
    )

    print("\nSearching only bucket_1:")

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Text: {result['text']}")
        print(f"Bucket: {result['metadata']['bucket_id']}")

    print("\n" + "=" * 70)
    print("CHROMADB TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()