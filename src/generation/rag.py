"""
End-to-end RAG generation pipeline.

Searches both knowledge buckets automatically and generates
an answer using the most relevant evidence.
"""

from typing import Dict, Any, List

from src.config.settings import (
    RERANK_RELEVANCE_THRESHOLD,
    BUCKETS,
    TOP_K,
)

from src.retrieval.retriever import Retriever
from src.generation.prompt import build_prompt
from src.generation.ollama_client import generate


NOT_FOUND_MESSAGE = (
    "Relevant information was not found in the knowledge base."
)


retriever = Retriever()


# ============================================================
# FILTER RELEVANT CHUNKS
# ============================================================

def filter_relevant_chunks(
    retrieved_chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Keep only strong and relevant chunks for generation.

    Rules:
    - Remove negative rerank scores
    - Keep chunks reasonably close to the best result
    - Return only the strongest chunks
    """

    if not retrieved_chunks:
        return []

    best_score = retrieved_chunks[0].get(
        "rerank_score",
        0.0,
    )

    filtered = []

    for chunk in retrieved_chunks:

        score = chunk.get(
            "rerank_score",
            0.0,
        )

        # Reject irrelevant chunks
        if score < 0:
            continue

        # Keep chunks reasonably close to best result
        if score >= best_score * 0.35:
            filtered.append(chunk)

    # Safety fallback
    if not filtered and best_score >= RERANK_RELEVANCE_THRESHOLD:
        filtered.append(retrieved_chunks[0])

    return filtered[:top_k]


# ============================================================
# SEARCH BOTH BUCKETS
# ============================================================

def retrieve_from_all_buckets(
    query: str,
) -> List[Dict[str, Any]]:
    """
    Search all configured buckets.

    Each bucket is retrieved independently because the Retriever
    performs bucket-specific dense retrieval and BM25 indexing.

    Results are then combined and sorted globally using rerank score.
    """

    all_chunks: List[Dict[str, Any]] = []

    for bucket_id in BUCKETS.keys():

        bucket_results = retriever.retrieve(
            question=query,
            bucket_id=bucket_id,
        )

        if bucket_results:
            all_chunks.extend(bucket_results)

    # Sort all bucket results globally
    all_chunks.sort(
        key=lambda chunk: chunk.get(
            "rerank_score",
            float("-inf"),
        ),
        reverse=True,
    )

    return all_chunks


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def answer_query(
    query: str,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """
    Answer a user question by searching both knowledge buckets.

    Pipeline:

        Question
            ↓
        Search Bucket 1
            +
        Search Bucket 2
            ↓
        Combine results
            ↓
        Global relevance ranking
            ↓
        Filter strong evidence
            ↓
        Build grounded prompt
            ↓
        Ollama generation
            ↓
        Final answer + sources
    """

    # ---------------------------------------------------------
    # STEP 1: Validate query
    # ---------------------------------------------------------

    if not query or not query.strip():

        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # ---------------------------------------------------------
    # STEP 2: Validate top_k
    # ---------------------------------------------------------

    if not top_k or top_k <= 0:
        top_k = TOP_K

    # Limit generation context
    generation_limit = min(top_k, 3)

    # ---------------------------------------------------------
    # STEP 3: Search both buckets
    # ---------------------------------------------------------

    retrieved_chunks = retrieve_from_all_buckets(
        query=query,
    )

    if not retrieved_chunks:

        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # ---------------------------------------------------------
    # STEP 4: Global relevance check
    # ---------------------------------------------------------

    best_rerank_score = retrieved_chunks[0].get(
        "rerank_score"
    )

    if (
        best_rerank_score is None
        or best_rerank_score < RERANK_RELEVANCE_THRESHOLD
    ):

        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # ---------------------------------------------------------
    # STEP 5: Filter strongest chunks
    # ---------------------------------------------------------

    generation_chunks = filter_relevant_chunks(
        retrieved_chunks=retrieved_chunks,
        top_k=generation_limit,
    )

    if not generation_chunks:

        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # ---------------------------------------------------------
    # STEP 6: Build grounded prompt
    # ---------------------------------------------------------

    prompt = build_prompt(
        query=query,
        retrieved_chunks=generation_chunks,
    )

    # ---------------------------------------------------------
    # STEP 7: Generate final answer
    # ---------------------------------------------------------

    answer = generate(
        prompt=prompt,
    )

    # Safe fallback
    if not answer or not answer.strip():

        answer = NOT_FOUND_MESSAGE

    # ---------------------------------------------------------
    # STEP 8: Prepare sources
    # ---------------------------------------------------------

    sources: List[Dict[str, Any]] = []

    for chunk in generation_chunks:

        metadata = chunk.get(
            "metadata",
            {},
        )

        sources.append(
            {
                "chunk_id": chunk.get("id"),
                "file_name": metadata.get("file_name"),
                "bucket_id": metadata.get("bucket_id"),
                "content_type": metadata.get("content_type"),
                "page_number": metadata.get("page_number"),
                "rerank_score": chunk.get("rerank_score"),
            }
        )

    return {
        "answer": answer.strip(),
        "sources": sources,
    }