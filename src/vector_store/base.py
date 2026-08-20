"""
Abstract interface for a vector store backend. retrieval/ and
scripts/ingest.py should only ever talk to this interface - never
import chroma_store.py or qdrant_store.py directly (use
vector_store/__init__.py's get_vector_store() factory instead).

This is what lets us start with Chroma (per spec, for limited local
resources) and switch to Qdrant later without rewriting retrieval logic.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStore(ABC):
    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add a batch of chunks (with their vectors + metadata) to the store."""
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        bucket_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top_k most similar chunks, optionally filtered to a
        single bucket_id. Each result should include at least:
        text, metadata, score.
        """
        raise NotImplementedError
    
    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        """Return True if the document has already been ingested."""
        raise NotImplementedError
    
    @abstractmethod
    def count(self, bucket_id: Optional[str] = None) -> int:
        """Number of chunks currently stored (optionally scoped to a bucket)."""
        raise NotImplementedError
