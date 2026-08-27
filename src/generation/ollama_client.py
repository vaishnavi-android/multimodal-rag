import ollama


OLLAMA_MODEL = "qwen3:8b"


def generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Generate a final answer using the local Ollama model.
    """

    if not prompt or not prompt.strip():
        return ""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise RAG answer generator. "
                        "Use only the provided evidence. "
                        "Return only the final answer."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.1,
                "num_predict": 512,
                "num_ctx": 8192,
            },
        )

        message = response.get("message", {})

        content = message.get(
            "content",
            "",
        ).strip()

        if content:
            return content

        response_text = response.get(
            "response",
            "",
        ).strip()

        if response_text:
            return response_text

        return ""

    except Exception as error:
        print(
            f"[GENERATION ERROR] {error}"
        )
        return ""