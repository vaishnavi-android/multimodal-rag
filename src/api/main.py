from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="Multimodal Bucket-Based RAG API",
    description="Bucket-aware retrieval-augmented generation over a multimodal document set.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "Multimodal RAG API is running. See /docs for endpoints."}
