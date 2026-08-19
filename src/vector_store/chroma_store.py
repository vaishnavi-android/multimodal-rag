"""
Chroma-backed implementation of VectorStore. Default backend per the
project spec (lightweight, fully local, no separate server process -
appropriate given limited local resources).

Requires: pip install chromadb
"""

from typing import List, Dict, Any, Optional

from src.vector_store.base import VectorStore
from src.config.settings import VECTOR_DB_DIR, COLLECTION_NAME


class ChromaStore(VectorStore):
    def __init__(self, persist_dir=VECTOR_DB_DIR, collection_name: str = COLLECTION_NAME):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(self, ids, embeddings, documents, metadatas) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        bucket_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        where = {"bucket_id": bucket_id} if bucket_id else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        out = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            out.append(
                {
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                }
            )
        return out

    def count(self, bucket_id: Optional[str] = None) -> int:
        if not bucket_id:
            return self._collection.count()

        results = self._collection.get(
            where={"bucket_id": bucket_id},
            include=[],
        )

        return len(results["ids"])
