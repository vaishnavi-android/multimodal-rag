
"""
Hybrid retriever.

Retrieval pipeline:

1. Dense semantic retrieval
2. BM25 keyword retrieval
3. Reciprocal Rank Fusion (RRF)
4. Cross-encoder reranking

The final results are filtered to the selected bucket.
"""

import re
from typing import Dict, List, Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config.settings import (
    DENSE_TOP_K,
    BM25_TOP_K,
    RRF_K,
    RERANK_CANDIDATES,
    FINAL_TOP_K,
)

from src.embeddings.embedder import embed_query
from src.vector_store import get_vector_store


class Retriever:
    """
    Hybrid retrieval system.

    Combines:
        - Dense semantic retrieval
        - BM25 keyword retrieval
        - Reciprocal Rank Fusion
        - Cross-encoder reranking
    """

    def __init__(self):
        """
        Initialize the vector store and reranker.
        """

        self.vector_store = get_vector_store()

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # Cache BM25 indexes separately for each bucket
        self._bm25_cache = {}

    # ============================================================
    # PUBLIC RETRIEVAL METHOD
    # ============================================================

    def retrieve(
        self,
        question: str,
        bucket_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a question.

        Pipeline:

            Question
                ↓
        Dense retrieval + BM25 retrieval
                ↓
            RRF fusion
                ↓
            Reranking
                ↓
            Final results
        """

        if not question or not question.strip():
            return []

        if not bucket_id:
            return []

        # --------------------------------------------------------
        # STEP 1: Dense semantic retrieval
        # --------------------------------------------------------

        dense_results = self._dense_retrieve(
            question=question,
            bucket_id=bucket_id,
        )

        # --------------------------------------------------------
        # STEP 2: BM25 keyword retrieval
        # --------------------------------------------------------

        bm25_results = self._bm25_retrieve(
            question=question,
            bucket_id=bucket_id,
        )

        # --------------------------------------------------------
        # STEP 3: RRF fusion
        # --------------------------------------------------------

        fused_results = self._rrf_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        # Limit candidates before reranking
        candidates = fused_results[:RERANK_CANDIDATES]

        # --------------------------------------------------------
        # STEP 4: Cross-encoder reranking
        # --------------------------------------------------------

        reranked_results = self._rerank(
            question=question,
            candidates=candidates,
        )

        # --------------------------------------------------------
        # FINAL TOP RESULTS
        # --------------------------------------------------------

        return reranked_results[:FINAL_TOP_K]

    # ============================================================
    # DENSE RETRIEVAL
    # ============================================================

    def _dense_retrieve(
        self,
        question: str,
        bucket_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic vector retrieval.
        """

        query_embedding = embed_query(question)

        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=DENSE_TOP_K,
            bucket_id=bucket_id,
        )

        output = []

        for result in results:
            output.append(
                {
                    "id": result["id"],
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "distance": result.get("distance"),
                    "retrieval_source": "dense",
                }
            )

        return output

    # ============================================================
    # BM25 RETRIEVAL
    # ============================================================

    def _bm25_retrieve(
        self,
        question: str,
        bucket_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword retrieval.
        """

        bm25, documents = self._get_bm25_index(
            bucket_id=bucket_id
        )

        if not documents:
            return []

        query_tokens = self._tokenize(question)

        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:BM25_TOP_K]:

            # Skip chunks with no keyword relevance
            if scores[index] <= 0:
                continue

            document = documents[index]

            results.append(
                {
                    "id": document["id"],
                    "text": document["text"],
                    "metadata": document["metadata"],
                    "bm25_score": float(scores[index]),
                    "retrieval_source": "bm25",
                }
            )

        return results

    # ============================================================
    # BM25 INDEX
    # ============================================================

    def _get_bm25_index(
        self,
        bucket_id: str,
    ):
        """
        Build and cache a BM25 index for a bucket.
        """

        if bucket_id in self._bm25_cache:
            return self._bm25_cache[bucket_id]

        documents = self.vector_store.get_documents(
            bucket_id=bucket_id
        )

        if not documents:
            empty_bm25 = BM25Okapi([[]])

            self._bm25_cache[bucket_id] = (
                empty_bm25,
                [],
            )

            return self._bm25_cache[bucket_id]

        tokenized_documents = [
            self._tokenize(document["text"])
            for document in documents
        ]

        bm25 = BM25Okapi(
            tokenized_documents
        )

        self._bm25_cache[bucket_id] = (
            bm25,
            documents,
        )

        return bm25, documents

    # ============================================================
    # RECIPROCAL RANK FUSION
    # ============================================================

    def _rrf_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine Dense and BM25 rankings using
        Reciprocal Rank Fusion.

        Formula:

            score = 1 / (RRF_K + rank)
        """

        fused = {}

        # Dense ranking
        for rank, result in enumerate(
            dense_results,
            start=1,
        ):

            chunk_id = result["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "sources": [],
                }

            fused[chunk_id]["rrf_score"] += (
                1 / (RRF_K + rank)
            )

            fused[chunk_id]["sources"].append(
                "dense"
            )

        # BM25 ranking
        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            chunk_id = result["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "sources": [],
                }

            fused[chunk_id]["rrf_score"] += (
                1 / (RRF_K + rank)
            )

            fused[chunk_id]["sources"].append(
                "bm25"
            )

        results = list(fused.values())

        results.sort(
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return results

    # ============================================================
    # RERANKING
    # ============================================================

    def _rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using a cross-encoder.

        The reranker receives:

            (question, chunk)

        together and produces a relevance score.
        """

        if not candidates:
            return []

        pairs = [
            (
                question,
                candidate["text"],
            )
            for candidate in candidates
        ]

        scores = self.reranker.predict(
            pairs
        )

        for candidate, score in zip(
            candidates,
            scores,
        ):

            candidate["rerank_score"] = float(
                score
            )

        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return candidates

    # ============================================================
    # TOKENIZATION
    # ============================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> List[str]:
        """
        Tokenize text for BM25 retrieval.

        - Converts text to lowercase
        - Removes punctuation
        - Keeps words and numbers
        """

        if not text:
            return []

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )


