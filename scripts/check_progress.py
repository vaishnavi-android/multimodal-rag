"""
Reports current document counts per bucket against the staged testing
progression (see README.md), so you always know exactly which stage
you're at without counting files by hand.

Usage:
    python scripts/check_progress.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import BUCKETS
from src.ingestion.file_detector import list_bucket_documents

STAGES = [
    ("Test 1", 1, "Prove basic ingestion -> retrieval"),
    ("Test 2", 2, "Different document types"),
    ("Test 3", 5, "Multiple chunks/documents"),
    ("Test 4", 10, "Bucket filtering properly"),
    ("Test 5", 25, "Stress retrieval"),
    ("Test 6", 50, "Complete one bucket"),
    ("Final", 100, "Full project"),
]


def current_stage(doc_count: int):
    reached = None
    next_stage = None
    for name, threshold, goal in STAGES:
        if doc_count >= threshold:
            reached = (name, threshold, goal)
        elif next_stage is None:
            next_stage = (name, threshold, goal)
    return reached, next_stage


def main():
    total = 0
    for bucket_id, bucket_dir in BUCKETS.items():
        count = len(list_bucket_documents(bucket_dir))
        total += count
        print(f"{bucket_id}: {count} document(s) in {bucket_dir}")

    print(f"\nTotal across both buckets: {total}")

    reached, next_stage = current_stage(total)

    if reached:
        print(f"\n✅ Currently at or past: {reached[0]} ({reached[1]} docs) - {reached[2]}")
    else:
        print("\n⚠️  Below Test 1 (need at least 1 document to start).")

    if next_stage:
        remaining = next_stage[1] - total
        print(f"➡️  Next: {next_stage[0]} ({next_stage[1]} docs) - {next_stage[2]}  "
              f"[{remaining} more document(s) needed]")
    else:
        print("🚀 Final stage (100 docs) reached.")


if __name__ == "__main__":
    main()
