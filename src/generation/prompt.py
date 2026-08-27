"""
Builds grounded prompts for the RAG generation stage.
"""

from typing import List, Dict, Any


NOT_FOUND_MESSAGE = (
    "Relevant information was not found in the knowledge base."
)


def build_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Build a grounded prompt for answer generation.

    The model must use only retrieved evidence and synthesize
    a natural answer instead of copying chunk text.
    """

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        text = chunk.get("text", "").strip()

        if not text:
            continue

        context_parts.append(
            f"""
EVIDENCE {index}:
{text}
""".strip()
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are the final answer generator for a Retrieval-Augmented
Generation system.

Answer the user's question using ONLY the evidence provided below.

YOUR JOB:

1. Read the evidence carefully.
2. Identify the facts that directly answer the question.
3. Combine relevant facts when necessary.
4. Write a fresh answer in natural language.
5. Do NOT copy an entire sentence or paragraph from the evidence.
6. Do NOT repeat the evidence.
7. Do NOT mention chunks, sources, retrieval, context,
   documents, or the knowledge base.
8. Do NOT add outside knowledge.
9. Use short and meaningful sentences.
10. Give a complete answer. Never stop in the middle of a sentence.
11. If the answer is clearly supported by the evidence,
    answer confidently.
12. Use this exact response only when NONE of the evidence
    contains enough information to answer:

"{NOT_FOUND_MESSAGE}"

IMPORTANT:

Do not summarize every piece of evidence.
Answer only what the user asked.

QUESTION:
{query}

EVIDENCE:
{context}

Now write the final answer only.
"""

    return prompt.strip()