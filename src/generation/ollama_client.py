import ollama


OLLAMA_MODEL = "qwen3:8b"


def generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Generate an answer using a locally running Ollama model.
    """

    if not prompt or not prompt.strip():
        return ""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0,
        },
    )

    message = response.get("message")

    if not message:
        return ""

    content = message.get("content")

    if not content:
        return ""

    return content.strip()