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


# ---------------------------------------------------------------------------
# Retrieval / relevance configuration
# ---------------------------------------------------------------------------

# Number of candidates retrieved from Chroma before reranking.
CANDIDATE_K = TOP_K

# Maximum number of evidence chunks allowed to reach the generation layer.
MAX_RELEVANT_CHUNKS = 8

# A negative best CrossEncoder score means the model did not find
# sufficiently relevant evidence.
MIN_BEST_SCORE = 0.0

# Maximum score difference allowed between the strongest evidence
# and additional supporting evidence.
#
# This is NOT a dataset-specific threshold.
# It controls how far down the CrossEncoder ranking we are willing
# to go from the strongest evidence.
MAX_SCORE_DROP = 2.5


# ---------------------------------------------------------------------------
# CrossEncoder
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_reranker():
    """
    Load the CrossEncoder once and reuse it.

    The model evaluates:

        (user query, candidate chunk)

    and produces a relevance score.
    """

    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _deduplicate_results(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove duplicate chunk text while preserving relevance order.
    """

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


# ---------------------------------------------------------------------------
# CrossEncoder reranking
# ---------------------------------------------------------------------------

def _rerank(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Rerank Chroma candidates using the CrossEncoder.

    Higher CrossEncoder scores indicate stronger query/chunk relevance.
    """

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

        # Preserve original Chroma distance.
        result["vector_distance"] = result.get("distance")

        # Store CrossEncoder relevance score.
        result["relevance_score"] = float(score)

        ranked.append(result)

    # Strongest relevance first.
    ranked.sort(
        key=lambda item: item["relevance_score"],
        reverse=True,
    )

    return ranked


# ---------------------------------------------------------------------------
# Evidence selection
# ---------------------------------------------------------------------------

def _select_relevant_evidence(
    ranked: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Select only evidence that is strongly connected to the query.

    The CrossEncoder ranking is used to determine whether the knowledge
    base actually contains useful evidence.

    Logic:

        1. No ranked results → no evidence.
        2. Negative best score → no evidence.
        3. Keep the strongest result.
        4. Keep additional results only when their score is close enough
           to the strongest result.
        5. Stop when the relevance score drops too far.

    This prevents weak, merely topical chunks from being sent to the LLM.
    """

    if not ranked:
        return []

    best_score = ranked[0].get(
        "relevance_score",
        float("-inf"),
    )

    # ---------------------------------------------------------------
    # No sufficiently relevant evidence exists.
    # ---------------------------------------------------------------

    if best_score < MIN_BEST_SCORE:
        return []

    selected = []

    for result in ranked:

        score = result.get(
            "relevance_score",
            float("-inf"),
        )

        score_drop = best_score - score

        # Stop once relevance has dropped too far from the best evidence.
        if score_drop > MAX_SCORE_DROP:
            break

        selected.append(result)

        if len(selected) >= MAX_RELEVANT_CHUNKS:
            break

    return selected


# ---------------------------------------------------------------------------
# Public retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve grounded evidence from the complete knowledge base.

    Pipeline:

        Query
          ↓
        Embedding
          ↓
        Chroma top-K candidates
          ↓
        CrossEncoder reranking
          ↓
        Evidence selection
          ↓
        Deduplication
          ↓
        Final evidence

    Returns an empty list when the knowledge base does not contain
    sufficiently relevant evidence.

    rag.py uses an empty result to prevent the LLM from being called.
    """

    if not query or not query.strip():
        return []

    # ---------------------------------------------------------------
    # 1. Embed the user query
    # ---------------------------------------------------------------

    query_embedding = embed_query(query)

    # ---------------------------------------------------------------
    # 2. Search the complete knowledge base
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # 3. CrossEncoder reranking
    # ---------------------------------------------------------------

    ranked = _rerank(
        query=query,
        candidates=candidates,
    )

    # ---------------------------------------------------------------
    # 4. Evidence selection
    # ---------------------------------------------------------------

    relevant = _select_relevant_evidence(ranked)

    if not relevant:
        return []

    # ---------------------------------------------------------------
    # 5. Remove duplicate evidence
    # ---------------------------------------------------------------

    relevant = _deduplicate_results(relevant)

    # ---------------------------------------------------------------
    # 6. Final evidence limit
    # ---------------------------------------------------------------

    return relevant[:MAX_RELEVANT_CHUNKS]