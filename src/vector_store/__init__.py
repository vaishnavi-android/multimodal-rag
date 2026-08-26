"""Vector store factory."""

from src.config.settings import VECTOR_STORE_BACKEND


def get_vector_store():
    if VECTOR_STORE_BACKEND == "chroma":
        from src.vector_store.chroma_store import ChromaStore

        return ChromaStore()

    raise ValueError(
        f"Unknown VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND}"
    )