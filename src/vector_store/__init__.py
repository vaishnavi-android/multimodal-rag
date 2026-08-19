"""
Factory: get_vector_store() returns a VectorStore instance for whichever
backend is configured. This is the ONLY place that should import
ChromaStore or QdrantStore directly - every other module should call
get_vector_store() and program against the VectorStore interface.
"""

from src.config.settings import VECTOR_STORE_BACKEND


def get_vector_store():
    if VECTOR_STORE_BACKEND == "chroma":
        from src.vector_store.chroma_store import ChromaStore
        return ChromaStore()
    elif VECTOR_STORE_BACKEND == "qdrant":
        from src.vector_store.qdrant_store import QdrantStore
        return QdrantStore()
    else:
        raise ValueError(f"Unknown VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND}")
