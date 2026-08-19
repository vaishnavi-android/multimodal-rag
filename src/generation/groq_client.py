from dotenv import load_dotenv
import os

load_dotenv()
"""
Groq generation client for the multimodal RAG system.

The filename is kept as ollama_client.py for compatibility with the
existing generation pipeline, but generation is performed through Groq.

Requires:
    pip install groq

Environment variable:
    GROQ_API_KEY
"""

import os

from groq import Groq


GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.6-27b",
)


def generate(
    prompt: str,
    model: str = GROQ_MODEL,
    stream: bool = False,
) -> str:
    """
    Generate the final grounded answer using Groq Qwen.

    Reasoning is explicitly disabled so that only the final answer is
    returned to the RAG pipeline.
    """

    if not prompt or not prompt.strip():
        return ""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        reasoning_effort="none",
        max_completion_tokens=500,
        stream=stream,
    )

    if stream:
        chunks = []

        for chunk in response:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                chunks.append(content)

        return "".join(chunks).strip()

    if not response.choices:
        return ""

    content = response.choices[0].message.content

    if not content:
        return ""

    return content.strip()

