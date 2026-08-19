"""
Retrieval layer for the multimodal bucket-based RAG system.

Every query searches the complete knowledge base.
The user does not select a bucket.
"""

from typing import List, Dict, Any

from src.embeddings.embedder import embed_query
from src.vector_store import get_vector_store
from src.config.settings import TOP_K, RETRIEVAL_DISTANCE_THRESHOLD


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique_results = []

    for result in results:
        text = result.get("text", "").strip()

        if not text:
            continue

        normalized = _normalize_text(text)

        if normalized in seen:
            continue

        seen.add(normalized)
        unique_results.append(result)

    return unique_results


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    query_embedding = embed_query(query)

    store = get_vector_store()

    candidates = store.query(
        query_embedding=query_embedding,
        top_k=top_k,
        bucket_id=None,
    )

    relevant = []

    for result in candidates:
        distance = result.get("distance", float("inf"))

        if distance <= RETRIEVAL_DISTANCE_THRESHOLD:
            relevant.append(result)

    return _deduplicate_results(relevant)