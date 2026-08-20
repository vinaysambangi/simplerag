"""Central configuration for the RAG pipeline.

All runtime knobs live here or in the project root `.env` file.
Never hardcode secrets in code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------
TEXT_DIR = BASE_DIR / "data" / "text_files"
PDF_DIR = BASE_DIR / "data" / "pdf"

# IMPORTANT: Chroma's persistent store is SQLite + binary HNSW index files.
# SQLite does not play well with cloud-sync folders (OneDrive, Dropbox,
# Google Drive) — they can dehydrate files to cloud-only placeholders or
# lock/rename them mid-sync while Chroma is reading, which causes
# intermittent partial reads (queries silently returning 0 results on some
# runs but not others). Keep the vector store OUTSIDE a synced folder.
# Override with RAG_VECTOR_DIR if you need a specific location.
VECTOR_DIR = Path(
    os.environ.get(
        "RAG_VECTOR_DIR",
        Path.home() / ".rag_local_data" / "vector_store",
    )
)

# SQLite database for chat sessions / messages (not synced to cloud).
CHAT_DB_PATH = Path(
    os.environ.get("RAG_CHAT_DB", BASE_DIR / "data" / "chat" / "chat.db")
)

# Generated Q&A dataset used for training/evaluation datasets.
QA_DATASET_PATH = Path(
    os.environ.get("QA_DATASET_PATH", BASE_DIR / "data" / "qa_dataset.jsonl")
)

# ---------------------------------------------------------------------------
# Vector store backend
# ---------------------------------------------------------------------------
VECTOR_DB = os.environ.get("VECTOR_DB", "QDRANT").upper()
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "Kamakshi")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Provider: "openai" (text-embedding-3-large, best quality) or "local"
# (sentence-transformers model, free / offline).
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai").lower()
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "text-embedding-3-large" if EMBEDDING_PROVIDER == "openai" else "all-MiniLM-L6-v2",
)
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
EMBEDDING_MAX_RETRIES = int(os.environ.get("EMBEDDING_MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "5"))
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("DEFAULT_SCORE_THRESHOLD", "0.0"))

# Hybrid search: dense vectors + BM25 keyword scores fused with RRF, then
# re-ranked by a cross-encoder.
HYBRID_SEARCH = os.environ.get("HYBRID_SEARCH", "true").lower() in (
    "1",
    "true",
    "yes",
)
# How many candidates each branch returns before fusion/reranking.
HYBRID_TOP_K = int(os.environ.get("HYBRID_TOP_K", "50"))
# RRF constant (lower = more weight to top ranks).
RRF_K = int(os.environ.get("RRF_K", "60"))
# Per-query metadata filter applied to candidates, e.g. {"source": "filename.pdf"}
# or {"source": ["a.pdf", "b.pdf"]}. Empty = no filter.
METADATA_FILTER = {}

# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
RERANK_MODEL_NAME = os.environ.get(
    "RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# Minimum rerank score (0..1) for a chunk to be kept. 0 = keep everything.
RERANK_MIN_SCORE = float(os.environ.get("RERANK_MIN_SCORE", "0.0"))

# ---------------------------------------------------------------------------
# Generation / chat
# ---------------------------------------------------------------------------
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2048"))

# Number of past conversation turns (user + assistant = 1 turn) sent to the
# LLM as chat history. History is scoped to the current session only.
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "6"))

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
RESET_VECTOR_STORE = os.environ.get("RESET_VECTOR_STORE", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Agentic chunker tuning
CHUNK_MAX_SIZE = int(os.environ.get("CHUNK_MAX_SIZE", "1200"))
CHUNK_BATCH_SIZE = int(os.environ.get("CHUNK_BATCH_SIZE", "25"))
CHUNK_MIN_LENGTH = int(os.environ.get("CHUNK_MIN_LENGTH", "100"))

# Chunking strategy: "hybrid" (LLM topic boundaries, semantic fallback),
# "semantic" (embedding-similarity boundaries, no LLM), or "size" (plain).
CHUNK_STRATEGY = os.environ.get("CHUNK_STRATEGY", "hybrid").lower()
# Context prefix style added to every chunk so it is self-contained.
CHUNK_CONTEXT_PREFIX = os.environ.get("CHUNK_CONTEXT_PREFIX", "true").lower() in (
    "1",
    "true",
    "yes",
)

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
