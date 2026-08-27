"""
End-to-end RAG generation pipeline.
"""

from typing import Dict, Any, List

from src.config.settings import RERANK_RELEVANCE_THRESHOLD
from src.retrieval.retriever import Retriever
from src.generation.prompt import build_prompt
from src.generation.ollama_client import generate


NOT_FOUND_MESSAGE = (
    "Relevant information was not found in the knowledge base."
)


retriever = Retriever()


def filter_relevant_chunks(
    retrieved_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Keep only chunks that are reasonably relevant
    compared with the best reranked result.
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

        # Reject negative chunks
        if score < 0:
            continue

        # Keep chunks reasonably close to the best result
        if score >= best_score * 0.35:
            filtered.append(chunk)

    # Always keep at least the best chunk
    if not filtered:
        filtered.append(retrieved_chunks[0])

    # Maximum 3 strong chunks for generation
    return filtered[:3]


def answer_query(
    query: str,
    bucket_id: str,
) -> Dict[str, Any]:
    """
    Answer a user question using the selected knowledge bucket.
    """

    if not query or not query.strip():
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    if not bucket_id or not bucket_id.strip():
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # ---------------------------------------------------------
    # STEP 1: Retrieve relevant chunks
    # ---------------------------------------------------------

    retrieved_chunks = retriever.retrieve(
        question=query,
        bucket_id=bucket_id,
    )

    if not retrieved_chunks:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # ---------------------------------------------------------
    # STEP 2: Check retrieval relevance
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
    # STEP 3: Filter relevant chunks and build prompt
    # ---------------------------------------------------------

    generation_chunks = filter_relevant_chunks(
        retrieved_chunks
    )

    prompt = build_prompt(
        query=query,
        retrieved_chunks=generation_chunks,
    )

    # ---------------------------------------------------------
    # STEP 4: Generate answer
    # ---------------------------------------------------------

    answer = generate(
        prompt=prompt,
    )

    # Safe fallback if generation fails
    if not answer or not answer.strip():
        answer = NOT_FOUND_MESSAGE

    # ---------------------------------------------------------
    # STEP 5: Prepare sources
    # ---------------------------------------------------------

    sources: List[Dict[str, Any]] = []

    for chunk in generation_chunks:
        metadata = chunk.get("metadata", {})

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
        "answer": answer,
        "sources": sources,
    }