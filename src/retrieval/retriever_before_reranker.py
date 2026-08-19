"""
Retrieval layer for the multimodal bucket-based RAG system.

Every query searches the complete knowledge base.
The user does not select a bucket.

This stage retrieves candidate chunks from the vector database.
Relevance filtering is handled separately so retrieval does not
depend on a fixed global distance threshold.
"""

from typing import List, Dict, Any

from src.embeddings.embedder import embed_query
from src.vector_store import get_vector_store
from src.config.settings import TOP_K


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
    """
    Retrieve candidate chunks from the complete knowledge base.

    Chroma already returns candidates ordered by vector similarity.
    We intentionally do not apply a fixed distance threshold here.
    A separate relevance layer will decide which candidates are
    sufficiently relevant for answering the query.
    """

    if not query or not query.strip():
        return []

    query_embedding = embed_query(query)

    store = get_vector_store()

    candidates = store.query(
        query_embedding=query_embedding,
        top_k=top_k,
        bucket_id=None,
    )

    return _deduplicate_results(candidates)