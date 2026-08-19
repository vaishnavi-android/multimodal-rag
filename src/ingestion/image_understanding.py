"""
Visual understanding for images (charts, diagrams, photos).

Uses Groq's vision-capable Qwen model instead of Ollama/LLaVA.
The image is sent as a base64 data URL and converted into a concise
text description that can be passed through the normal RAG pipeline.

Requires:
    pip install groq

Environment variable:
    GROQ_API_KEY
"""

import base64
import io
import os

from groq import Groq


VISION_MODEL = "qwen/qwen3.6-27b"

DESCRIBE_PROMPT = (
    "Analyze this image for a multimodal RAG system. "
    "Return ONLY the final factual answer, with no reasoning or <think> tags. "
    "In 1-2 short sentences, describe the image and transcribe important "
    "visible text exactly. Do not speculate."
)


def _image_to_base64(image) -> str:
    """Convert a PIL image to a base64-encoded PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def describe_image(image, model: str = VISION_MODEL) -> str:
    """
    Return a short textual description of an image using Groq vision.

    If Groq is unavailable or the API request fails, return an empty
    string so OCR can still provide useful content.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            print("[image_understanding] GROQ_API_KEY is not set")
            return ""

        encoded = _image_to_base64(image)

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": DESCRIBE_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded}"
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=800,
        )

        content = response.choices[0].message.content.strip()

        # Remove model reasoning if it is returned in <think>...</think> tags.
        if "<think>" in content:
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            else:
                # Reasoning was truncated before the closing tag.
                content = ""

        return content

    except Exception as e:
        print(f"[image_understanding] vision description skipped: {e}")
        return ""