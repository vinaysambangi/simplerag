"""
Shared configuration for the RAG pipeline.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TEXT_DIR = BASE_DIR / "data" / "text_files"
PDF_DIR = BASE_DIR / "data" / "pdf"
VECTOR_DIR = BASE_DIR / "data" / "vector_store"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "Kamakshi"

DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.0

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024