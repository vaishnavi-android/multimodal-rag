from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        examples=["Which organisms have life cycle posters?"],
    )
    top_k: Optional[int] = None


class SourceInfo(BaseModel):
    file_name: str
    page_number: Optional[int] = None
    bucket_id: str
    content_type: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]


class IngestRequest(BaseModel):
    bucket_id: Optional[str] = None
    # None = ingest both buckets


class IngestResponse(BaseModel):
    documents_processed: int
    chunks_created: int
    bucket_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str

