import ollama


OLLAMA_MODEL = "qwen3:8b"


def generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
) -> str:

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
            "num_predict": 300,
            "num_ctx": 4096,
        },
    )

    message = response.get("message", {})
    content = message.get("content", "")

    return content.strip()