"""
End-to-end RAG generation.
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
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Answer a user question using the complete knowledge base.

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

    # Retrieve relevant evidence
    retrieved_chunks = retrieve(
        query=query,
        top_k=top_k,
    )

    # Do not call the LLM when no evidence is found
    if not retrieved_chunks:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    # Build grounded prompt and generate answer
    prompt = build_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    answer = generate(prompt)

    # Prepare source metadata
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