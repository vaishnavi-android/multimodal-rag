"""
Tests vector database retrieval.
"""

import sys
from pathlib import Path


# Allow running:
# python scripts/test_retrieval.py
# from the project root.

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)


from src.retrieval.retriever import retrieve


def main():

    # ---------------------------------------------------------
    # TEST QUESTION
    # ---------------------------------------------------------

    question = "What is the capital of Karnataka?"

    # ---------------------------------------------------------
    # SELECT BUCKET
    # ---------------------------------------------------------

    bucket_id = "bucket_1"

    # ---------------------------------------------------------
    # RETRIEVE
    # ---------------------------------------------------------

    results = retrieve(
        query=question,
        bucket_id=bucket_id,
        top_k=5,
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL TEST")
    print("=" * 70)

    print(f"\nQuestion: {question}")
    print(f"Bucket: {bucket_id}")
    print(f"Results found: {len(results)}")

    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------

    for i, result in enumerate(results, start=1):

        print("\n" + "-" * 70)

        print(f"RESULT {i}")

        print("-" * 70)

        print(
            f"Distance: "
            f"{result['distance']}"
        )

        print("\nMetadata:")

        for key, value in result["metadata"].items():
            print(
                f"  {key}: {value}"
            )

        print("\nRetrieved Text:")

        print(
            result["text"]
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()