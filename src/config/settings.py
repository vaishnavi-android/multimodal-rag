"""Central configuration for the Multimodal Bucket-Based RAG system."""

import os
from pathlib import Path


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
BUCKET_1_DIR = DATA_DIR / "bucket_1"
BUCKET_2_DIR = DATA_DIR / "bucket_2"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
LOG_DIR = BASE_DIR / "logs"

BUCKETS = {
    "bucket_1": BUCKET_1_DIR,
    "bucket_2": BUCKET_2_DIR,
}

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))


# Embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", 384)
)

# Vector store
VECTOR_STORE_BACKEND = os.getenv(
    "VECTOR_STORE_BACKEND",
    "chroma",
)
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "multimodal_rag",
)


# Retrieval
TOP_K = int(os.getenv("TOP_K", 10))

RETRIEVAL_DISTANCE_THRESHOLD = float(
    os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", 1.0)
)


# OCR
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract")
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")


# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))