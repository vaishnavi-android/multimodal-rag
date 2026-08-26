"""
Wraps the embedding model
"""

from functools import lru_cache
from typing import List

from src.config.settings import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts. Used for chunk embedding during ingestion."""
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single query string at retrieval time."""
    return embed_texts([text])[0]
