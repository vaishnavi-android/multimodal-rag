"""
Retrieval + relevance reranking layer for the multimodal RAG system.

Flow:

    User query
        ↓
    Query embedding
        ↓
    Chroma vector search
        ↓
    Top-K candidate chunks
        ↓
    CrossEncoder reranking
        ↓
    Evidence selection
        ↓
    Best evidence chunks
"""

from functools import lru_cache
from typing import List, Dict, Any

from src.embeddings.embedder import embed_query
from src.vector_store import get_vector_store
from src.config.settings import TOP_K


CANDIDATE_K = TOP_K
MAX_RELEVANT_CHUNKS = 4
MIN_BEST_SCORE = 0.0
MAX_SCORE_DROP = 2.5


@lru_cache(maxsize=1)
def _get_reranker():
    """Load and cache the CrossEncoder reranker."""

    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _deduplicate_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove duplicate chunk text while preserving order."""

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


def _rerank(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rerank retrieved chunks using a CrossEncoder."""

    if not candidates:
        return []

    reranker = _get_reranker()

    pairs = [
        (query, candidate.get("text", ""))
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked = []

    for candidate, score in zip(candidates, scores):
        result = dict(candidate)

        result["vector_distance"] = result.get("distance")
        result["relevance_score"] = float(score)

        ranked.append(result)

    ranked.sort(
        key=lambda item: item["relevance_score"],
        reverse=True,
    )

    return ranked


def _select_relevant_evidence(
    ranked: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select the strongest relevant evidence."""

    if not ranked:
        return []

    best_score = ranked[0].get(
        "relevance_score",
        float("-inf"),
    )

    if best_score < MIN_BEST_SCORE:
        return []

    selected = []

    for result in ranked:
        score = result.get(
            "relevance_score",
            float("-inf"),
        )

        score_drop = best_score - score

        # Stop when relevance drops too far.
        if score_drop > MAX_SCORE_DROP:
            break

        selected.append(result)

        if len(selected) >= MAX_RELEVANT_CHUNKS:
            break

    return selected


def retrieve(
    query: str,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """Retrieve and rerank relevant evidence from the knowledge base."""

    if not query or not query.strip():
        return []

    query_embedding = embed_query(query)
    store = get_vector_store()

    candidate_count = max(
        top_k,
        CANDIDATE_K,
    )

    candidates = store.query(
        query_embedding=query_embedding,
        top_k=candidate_count,
        bucket_id=None,
    )

    if not candidates:
        return []

    ranked = _rerank(
        query=query,
        candidates=candidates,
    )

    relevant = _select_relevant_evidence(ranked)

    if not relevant:
        return []

    relevant = _deduplicate_results(relevant)

    return relevant[:MAX_RELEVANT_CHUNKS]