"""
Central configuration for the Multimodal Bucket-Based RAG system.

Every tunable value lives here (sourced from environment variables / .env
where possible). Pipeline modules should import from this file rather than
hard-coding chunk sizes, model names, ports, etc.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed yet - fine, real env vars still work.
    pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
BUCKET_1_DIR = DATA_DIR / "bucket_1"
BUCKET_2_DIR = DATA_DIR / "bucket_2"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
LOG_DIR = BASE_DIR / "logs"

# Single source of truth for valid bucket ids -> their folder on disk.
BUCKETS = {
    "bucket_1": BUCKET_1_DIR,
    "bucket_2": BUCKET_2_DIR,
}

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))  # overlap between chunks

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chroma")  # "chroma" | "qdrant"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multimodal_rag")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", 40))
RETRIEVAL_DISTANCE_THRESHOLD = float(
    os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", 1.0)
)
# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
# RapidOCR handles the OCR engine internally.

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
