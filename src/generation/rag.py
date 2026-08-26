"""
End-to-end RAG generation.

Flow:

    User question
        ↓
    Retrieval
        ↓
    Relevance filtering / deduplication
        ↓
    Grounded prompt
        ↓
    Ollama LLM
        ↓
    Final answer
"""

from typing import Dict, Any

from src.retrieval.retriever import retrieve
from src.generation.prompt import build_prompt
from src.generation.ollama_client import generate


NOT_FOUND_MESSAGE = (
    "Relevant information was not found in the knowledge base."
)


def answer_query(
    query: str,
    top_k: int = 40,
) -> Dict[str, Any]:
    """
    Answer a user question using the complete knowledge base.

    The user does not select a bucket. Retrieval searches across all
    configured buckets.

    Returns:
        {
            "answer": "...",
            "sources": [...],
        }
    """

    if not query or not query.strip():
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # ---------------------------------------------------------------
    # 1. Retrieve evidence from the complete knowledge base
    # ---------------------------------------------------------------
    retrieved_chunks = retrieve(
        query=query,
        top_k=top_k,
    )

    # ---------------------------------------------------------------
    # 2. If nothing relevant was retrieved, do not call the LLM.
    # ---------------------------------------------------------------
    if not retrieved_chunks:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # ---------------------------------------------------------------
    # 3. Build grounded prompt
    # ---------------------------------------------------------------
    prompt = build_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    # ---------------------------------------------------------------
    # 4. Generate answer with Ollama
    # ---------------------------------------------------------------
    answer = generate(prompt)

    # ---------------------------------------------------------------
    # 5. Return answer + source metadata
    # ---------------------------------------------------------------
    sources = []

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})

        sources.append(
            {
                "file_name": metadata.get("file_name"),
                "bucket_id": metadata.get("bucket_id"),
                "content_type": metadata.get("content_type"),
                "page_number": metadata.get("page_number"),
                "distance": chunk.get("distance"),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }