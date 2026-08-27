from src.generation.rag import answer_query


TEST_CASES = [
    {
        "name": "Exact factual question",
        "question": "What is the capital of Karnataka?",
        "bucket": "bucket_1",
    },
    {
        "name": "Keyword-based question",
        "question": "Which rivers are important in Karnataka?",
        "bucket": "bucket_1",
    },
    {
        "name": "Semantic question",
        "question": "Which city is known as the technology hub of Karnataka?",
        "bucket": "bucket_1",
    },
    {
        "name": "Topic question",
        "question": "What are the major industries in Kerala?",
        "bucket": "bucket_1",
    },
    {
        "name": "Unknown question",
        "question": "What is the population of Mars?",
        "bucket": "bucket_1",
    },
]


def main():

    print("\n" + "=" * 70)
    print("END-TO-END RAG TEST")
    print("=" * 70)

    for index, test in enumerate(TEST_CASES, start=1):

        print("\n" + "=" * 70)
        print(f"TEST CASE {index}")
        print("=" * 70)

        print(f"\nTest Type: {test['name']}")
        print(f"Question: {test['question']}")
        print(f"Bucket: {test['bucket']}")

        print("\nProcessing...")

        result = answer_query(
            query=test["question"],
            bucket_id=test["bucket"],
        )

        print("\n" + "-" * 70)
        print("FINAL ANSWER")
        print("-" * 70)

        print(result["answer"])

        print("\n" + "-" * 70)
        print("SOURCES")
        print("-" * 70)

        sources = result.get("sources", [])

        if not sources:
            print("No sources found.")

        else:
            for source_index, source in enumerate(
                sources,
                start=1,
            ):

                print(f"\nSOURCE {source_index}")

                for key, value in source.items():
                    print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("ALL END-TO-END RAG TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()