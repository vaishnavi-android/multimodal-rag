"""
Central logging configuration for the RAG pipeline.
"""

import logging
from pathlib import Path


# ---------------------------------------------------------
# Create logs directory
# ---------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Log file
# ---------------------------------------------------------

LOG_FILE = LOG_DIR / "rag_pipeline.log"


# ---------------------------------------------------------
# Configure logger
# ---------------------------------------------------------

logger = logging.getLogger("rag_pipeline")

logger.setLevel(logging.INFO)


# Avoid duplicate handlers when FastAPI reloads
if not logger.handlers:

    # File handler
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    # Console handler
    console_handler = logging.StreamHandler()

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)