from fastapi import APIRouter

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceInfo,
    IngestRequest,
    IngestResponse,
    HealthResponse,
)
from src.generation.rag import answer_query
from src.config.settings import BUCKETS, TOP_K

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = answer_query(
        query=request.query,
        top_k=request.top_k or TOP_K,
    )

    sources = [
        SourceInfo(
            file_name=source.get("file_name", "unknown"),
            page_number=source.get("page_number"),
            bucket_id=source.get("bucket_id", "unknown"),
            content_type=source.get("content_type", "text"),
        )
        for source in result.get("sources", [])
    ]

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    from scripts.ingest import run_ingestion

    bucket_ids = (
        [request.bucket_id]
        if request.bucket_id
        else list(BUCKETS.keys())
    )

    total_docs = 0
    total_chunks = 0

    for bucket_id in bucket_ids:
        docs, chunks = run_ingestion(bucket_id)
        total_docs += docs
        total_chunks += chunks

    return IngestResponse(
        documents_processed=total_docs,
        chunks_created=total_chunks,
        bucket_id=request.bucket_id,
    )

