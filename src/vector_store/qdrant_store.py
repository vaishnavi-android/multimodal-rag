"""
Qdrant-backed implementation of VectorStore. Not needed for Phase 1-4
(we start with Chroma per spec), but the interface is defined now so
switching backends later is a one-line change in
vector_store/__init__.py's factory - no changes needed in retrieval/.

Requires: pip install qdrant-client
"""

from typing import List, Dict, Any, Optional

from src.vector_store.base import VectorStore
from src.config.settings import COLLECTION_NAME


class QdrantStore(VectorStore):
    def __init__(self, url: str = "http://localhost:6333", collection_name: str = COLLECTION_NAME):
        raise NotImplementedError(
            "QdrantStore is a placeholder for a later phase. "
            "Use ChromaStore (VECTOR_STORE_BACKEND=chroma) for now."
        )

    def add(self, ids, embeddings, documents, metadatas) -> None:
        raise NotImplementedError

    def query(self, query_embedding, top_k, bucket_id=None):
        raise NotImplementedError

    def count(self, bucket_id: Optional[str] = None) -> int:
        raise NotImplementedError
