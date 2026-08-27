
"""
End-to-end RAG generation pipeline.

"""

from typing import Dict, Any, List

from src.config.settings import (
    RERANK_RELEVANCE_THRESHOLD,
    BUCKETS,
    TOP_K,
)

from src.retrieval.retriever import Retriever
from src.generation.prompt import build_prompt
from src.generation.ollama_client import generate
from src.utils.logger import logger


NOT_FOUND_MESSAGE = (
    "Relevant information was not found in the knowledge base."
)


retriever = Retriever()


# ============================================================
# FILTER RELEVANT CHUNKS
# ============================================================

def filter_relevant_chunks(
    retrieved_chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Keep only strong and relevant chunks for generation.

    Rules:
    - Remove negative rerank scores
    - Keep chunks reasonably close to the best result
    - Return only the strongest chunks
    """

    if not retrieved_chunks:

        logger.warning(
            "No retrieved chunks available for relevance filtering."
        )

        return []

    best_score = retrieved_chunks[0].get(
        "rerank_score",
        0.0,
    )

    logger.info(
        "Filtering retrieved chunks. "
        f"Total chunks: {len(retrieved_chunks)}, "
        f"best rerank score: {best_score}, "
        f"requested top_k: {top_k}"
    )

    filtered = []

    for chunk in retrieved_chunks:

        score = chunk.get(
            "rerank_score",
            0.0,
        )

        chunk_id = chunk.get(
            "id",
            "unknown_chunk",
        )

        # Reject irrelevant chunks
        if score < 0:

            logger.info(
                f"Rejected chunk {chunk_id}: "
                f"negative rerank score = {score}"
            )

            continue

        # Keep chunks reasonably close to best result
        if score >= best_score * 0.35:

            filtered.append(chunk)

            logger.info(
                f"Accepted chunk {chunk_id}: "
                f"rerank score = {score}"
            )

        else:

            logger.info(
                f"Rejected chunk {chunk_id}: "
                f"rerank score {score} below "
                f"relative relevance threshold."
            )

    # Safety fallback
    if (
        not filtered
        and best_score >= RERANK_RELEVANCE_THRESHOLD
    ):

        filtered.append(
            retrieved_chunks[0]
        )

        logger.info(
            "No chunks passed relative filtering. "
            "Using best chunk as safety fallback."
        )

    final_chunks = filtered[:top_k]

    logger.info(
        "Chunk filtering complete. "
        f"Chunks selected for generation: {len(final_chunks)}"
    )

    return final_chunks


# ============================================================
# SEARCH BOTH BUCKETS
# ============================================================

def retrieve_from_all_buckets(
    query: str,
) -> List[Dict[str, Any]]:
    """
    Search all configured buckets.

    Each bucket is retrieved independently because the Retriever
    performs bucket-specific dense retrieval and BM25 indexing.

    Results are then combined and sorted globally using rerank score.
    """

    all_chunks: List[Dict[str, Any]] = []

    logger.info(
        "Starting retrieval across all configured buckets."
    )

    for bucket_id in BUCKETS.keys():

        logger.info(
            f"Searching bucket: {bucket_id}"
        )

        bucket_results = retriever.retrieve(
            question=query,
            bucket_id=bucket_id,
        )

        result_count = (
            len(bucket_results)
            if bucket_results
            else 0
        )

        logger.info(
            f"Bucket {bucket_id} returned "
            f"{result_count} chunks."
        )

        if bucket_results:

            all_chunks.extend(
                bucket_results
            )

    logger.info(
        "Combining results from all buckets. "
        f"Total retrieved chunks: {len(all_chunks)}"
    )

    # Sort all bucket results globally
    all_chunks.sort(
        key=lambda chunk: chunk.get(
            "rerank_score",
            float("-inf"),
        ),
        reverse=True,
    )

    logger.info(
        "Global rerank sorting completed."
    )

    if all_chunks:

        logger.info(
            "Best global result: "
            f"chunk_id={all_chunks[0].get('id')}, "
            f"rerank_score="
            f"{all_chunks[0].get('rerank_score')}"
        )

    return all_chunks


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def answer_query(
    query: str,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """
    Answer a user question by searching both knowledge buckets.

    Pipeline:

        Question
            ↓
        Search Bucket 1
            +
        Search Bucket 2
            ↓
        Combine results
            ↓
        Global relevance ranking
            ↓
        Filter strong evidence
            ↓
        Build grounded prompt
            ↓
        Ollama generation
            ↓
        Final answer + sources
    """

    logger.info("=" * 70)
    logger.info("NEW RAG QUERY STARTED")
    logger.info(f"User query: {query}")
    logger.info(f"Requested top_k: {top_k}")

    # ---------------------------------------------------------
    # STEP 1: Validate query
    # ---------------------------------------------------------

    logger.info(
        "STEP 1: Validating user query."
    )

    if not query or not query.strip():

        logger.warning(
            "Query validation failed: empty query received."
        )

        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    logger.info(
        "Query validation successful."
    )

    # ---------------------------------------------------------
    # STEP 2: Validate top_k
    # ---------------------------------------------------------

    logger.info(
        "STEP 2: Validating top_k."
    )

    if not top_k or top_k <= 0:

        logger.warning(
            f"Invalid top_k={top_k}. "
            f"Using default TOP_K={TOP_K}."
        )

        top_k = TOP_K

    # Limit generation context
    generation_limit = min(
        top_k,
        3,
    )

    logger.info(
        f"Generation context limit set to "
        f"{generation_limit} chunks."
    )

    # ---------------------------------------------------------
    # STEP 3: Search both buckets
    # ---------------------------------------------------------

    logger.info(
        "STEP 3: Searching all knowledge buckets."
    )

    retrieved_chunks = retrieve_from_all_buckets(
        query=query,
    )

    if not retrieved_chunks:

        logger.warning(
            "No chunks were retrieved from any bucket."
        )

        logger.info(
            "Returning NOT_FOUND_MESSAGE."
        )

        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    logger.info(
        f"Retrieval completed successfully. "
        f"Total chunks retrieved: "
        f"{len(retrieved_chunks)}"
    )

    # ---------------------------------------------------------
    # STEP 4: Global relevance check
    # ---------------------------------------------------------

    logger.info(
        "STEP 4: Performing global relevance check."
    )

    best_rerank_score = retrieved_chunks[0].get(
        "rerank_score"
    )

    logger.info(
        f"Best rerank score: {best_rerank_score}. "
        f"Required threshold: "
        f"{RERANK_RELEVANCE_THRESHOLD}"
    )

    if (
        best_rerank_score is None
        or best_rerank_score < RERANK_RELEVANCE_THRESHOLD
    ):

        logger.warning(
            "Best retrieved result did not meet "
            "the relevance threshold."
        )

        logger.info(
            "Returning NOT_FOUND_MESSAGE."
        )

        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    logger.info(
        "Global relevance check passed."
    )

    # ---------------------------------------------------------
    # STEP 5: Filter strongest chunks
    # ---------------------------------------------------------

    logger.info(
        "STEP 5: Filtering strongest evidence chunks."
    )

    generation_chunks = filter_relevant_chunks(
        retrieved_chunks=retrieved_chunks,
        top_k=generation_limit,
    )

    if not generation_chunks:

        logger.warning(
            "No relevant chunks remained after filtering."
        )

        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
        }

    logger.info(
        f"Evidence filtering complete. "
        f"{len(generation_chunks)} chunks "
        f"selected for answer generation."
    )

    for index, chunk in enumerate(
        generation_chunks,
        start=1,
    ):

        metadata = chunk.get(
            "metadata",
            {},
        )

        logger.info(
            f"Generation source {index}: "
            f"chunk_id={chunk.get('id')}, "
            f"file={metadata.get('file_name')}, "
            f"bucket={metadata.get('bucket_id')}, "
            f"score={chunk.get('rerank_score')}"
        )

    # ---------------------------------------------------------
    # STEP 6: Build grounded prompt
    # ---------------------------------------------------------

    logger.info(
        "STEP 6: Building grounded generation prompt."
    )

    prompt = build_prompt(
        query=query,
        retrieved_chunks=generation_chunks,
    )

    logger.info(
        f"Prompt built successfully. "
        f"Prompt length: {len(prompt)} characters."
    )

    # ---------------------------------------------------------
    # STEP 7: Generate final answer
    # ---------------------------------------------------------

    logger.info(
        "STEP 7: Sending grounded prompt to Ollama "
        "for answer generation."
    )

    answer = generate(
        prompt=prompt,
    )

    # Safe fallback
    if not answer or not answer.strip():

        logger.warning(
            "Generation returned an empty answer. "
            "Using NOT_FOUND_MESSAGE fallback."
        )

        answer = NOT_FOUND_MESSAGE

    else:

        logger.info(
            "Answer generated successfully."
        )

        logger.info(
            f"Generated answer length: "
            f"{len(answer.strip())} characters."
        )

    # ---------------------------------------------------------
    # STEP 8: Prepare sources
    # ---------------------------------------------------------

    logger.info(
        "STEP 8: Preparing source metadata "
        "for API/frontend response."
    )

    sources: List[Dict[str, Any]] = []

    for chunk in generation_chunks:

        metadata = chunk.get(
            "metadata",
            {},
        )

        sources.append(
            {
                "chunk_id": chunk.get("id"),
                "file_name": metadata.get("file_name"),
                "bucket_id": metadata.get("bucket_id"),
                "content_type": metadata.get("content_type"),
                "page_number": metadata.get("page_number"),
                "rerank_score": chunk.get("rerank_score"),
            }
        )

    logger.info(
        f"Prepared {len(sources)} source records."
    )

    logger.info(
        "RAG QUERY COMPLETED SUCCESSFULLY."
    )

    logger.info("=" * 70)

    return {
        "answer": answer.strip(),
        "sources": sources,
    }

