"""
Prompt construction for the RAG generation layer.

The LLM must answer ONLY from evidence retrieved from the knowledge base.
It must not use outside knowledge or guess when the evidence is insufficient.
"""

from typing import List, Dict, Any


SYSTEM_INSTRUCTIONS = (
    "You are the answer-generation component of a Retrieval-Augmented "
    "Generation (RAG) system.\n\n"

    "You MUST answer the user's question using ONLY the information "
    "contained in the provided knowledge-base evidence.\n\n"

    "Rules:\n"
    "1. Do not use outside knowledge.\n"
    "2. Do not invent, assume, or hallucinate facts.\n"
    "3. Combine information from multiple sources when necessary.\n"
    "4. If the evidence contains enough information, give a clear and "
    "direct answer to the question.\n"
    "5. If the evidence does not contain enough relevant information, "
    "respond exactly with: "
    "\"Relevant information was not found in the knowledge base.\"\n"
    "6. Do not answer a question merely because the topic is vaguely "
    "related to the retrieved evidence.\n"
    "7. Keep the answer concise but complete.\n"
)


def build_context_block(
    retrieved_chunks: List[Dict[str, Any]]
) -> str:
    """
    Convert retrieved evidence into a clearly labelled context block.
    """

    if not retrieved_chunks:
        return ""

    blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {})

        source_label = meta.get("file_name", "unknown")

        page = meta.get("page_number")

        if page is not None:
            source_label = f"{source_label} (page {page})"

        bucket = meta.get("bucket_id")

        if bucket:
            source_label = f"{source_label}, {bucket}"

        blocks.append(
            f"[Evidence {i}: {source_label}]\n"
            f"{chunk.get('text', '').strip()}"
        )

    return "\n\n---\n\n".join(blocks)


def build_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> str:
    """
    Build the final grounded prompt sent to the language model.
    """

    context_block = build_context_block(retrieved_chunks)

    if not context_block:
        return (
            f"{SYSTEM_INSTRUCTIONS}\n\n"
            f"KNOWLEDGE-BASE EVIDENCE:\n"
            f"No relevant evidence was retrieved.\n\n"
            f"QUESTION:\n{query}\n\n"
            f"ANSWER:"
        )

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"KNOWLEDGE-BASE EVIDENCE:\n"
        f"{context_block}\n\n"
        f"QUESTION:\n"
        f"{query}\n\n"
        f"ANSWER:"
    )