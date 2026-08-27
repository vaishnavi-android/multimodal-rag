from src.retrieval.retriever import Retriever


TEST_CASES = [
    {
        "question": "What is the capital of Karnataka?",
        "bucket_id": "bucket_1",
        "test_type": "Exact factual question",
    },
    {
        "question": "Which rivers are important in Karnataka?",
        "bucket_id": "bucket_1",
        "test_type": "Keyword-based question",
    },
    {
        "question": "Which city is known as the technology hub of Karnataka?",
        "bucket_id": "bucket_1",
        "test_type": "Semantic / paraphrased question",
    },
    {
        "question": "What are the major industries in Kerala?",
        "bucket_id": "bucket_1",
        "test_type": "Topic retrieval question",
    },
    {
        "question": "What is the population of Mars?",
        "bucket_id": "bucket_1",
        "test_type": "Negative / unknown question",
    },
]


def print_result(result_number, result):
    print("\n" + "-" * 70)
    print(f"RESULT {result_number}")
    print("-" * 70)

    print("\nChunk ID:")
    print(result.get("id"))

    print("\nRRF Score:")
    print(result.get("rrf_score"))

    print("\nRerank Score:")
    print(result.get("rerank_score"))

    print("\nRetrieval Sources:")
    print(result.get("sources"))

    print("\nMetadata:")

    metadata = result.get("metadata", {})

    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print("\nRetrieved Text:")
    print(result.get("text", ""))


def main():

    print("\n" + "=" * 70)
    print("MULTI-QUESTION HYBRID RETRIEVAL EVALUATION")
    print("=" * 70)

    # Create retriever only once
    retriever = Retriever()

    for test_number, test in enumerate(TEST_CASES, start=1):

        question = test["question"]
        bucket_id = test["bucket_id"]
        test_type = test["test_type"]

        print("\n")
        print("=" * 70)
        print(f"TEST CASE {test_number}")
        print("=" * 70)

        print(f"\nTest Type: {test_type}")
        print(f"Question: {question}")
        print(f"Bucket: {bucket_id}")

        results = retriever.retrieve(
            question=question,
            bucket_id=bucket_id,
        )

        print(f"\nFinal Results Found: {len(results)}")

        if not results:
            print("\nNo results found.")
            continue

        for result_number, result in enumerate(results, start=1):
            print_result(
                result_number=result_number,
                result=result,
            )

        # --------------------------------------------------------
        # BUCKET VALIDATION
        # --------------------------------------------------------

        wrong_bucket_results = []

        for result in results:

            metadata = result.get("metadata", {})

            result_bucket = metadata.get("bucket_id")

            if result_bucket != bucket_id:
                wrong_bucket_results.append(result)

        print("\n" + "-" * 70)
        print("BUCKET VALIDATION")
        print("-" * 70)

        if wrong_bucket_results:
            print(
                f"FAILED: {len(wrong_bucket_results)} result(s) "
                "came from the wrong bucket."
            )
        else:
            print(
                "PASSED: All retrieved chunks belong "
                "to the selected bucket."
            )

    print("\n" + "=" * 70)
    print("ALL RETRIEVAL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()