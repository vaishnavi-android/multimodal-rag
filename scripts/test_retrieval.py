"""
Quick manual sanity check for Phase 5: prove bucket-filtered retrieval
works BEFORE wiring up Ollama, per the recommended build order.

Usage:
    python scripts/test_retrieval.py bucket_1 "What is the minimum age required?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import retrieve


def main():
    if len(sys.argv) < 3:
        print('Usage: python scripts/test_retrieval.py <bucket_id> "<query>"')
        sys.exit(1)

    bucket_id = sys.argv[1]
    query = sys.argv[2]

    results = retrieve(query=query, bucket_id=bucket_id)

    print(f"\nQuery: {query}")
    print(f"Bucket: {bucket_id}")
    print(f"Retrieved {len(results)} chunk(s):\n")

    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        print(f"--- Result {i} (score={r.get('score')}) ---")
        print(f"  file: {meta.get('file_name')}  page: {meta.get('page_number')}  bucket: {meta.get('bucket_id')}")
        print(f"  text: {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
