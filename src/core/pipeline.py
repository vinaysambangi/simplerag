"""Full RAG pipeline orchestration.

Stages: INGESTION -> CHUNKING -> EMBEDDING -> VECTOR STORE -> RETRIEVAL -> GENERATION

Components are created lazily and cached so the API server and scripts can
share one set of singletons (embedding model is ~100MB, avoid reloading it
per request).
"""

import os
from typing import Any, Dict, Generator, List, Tuple

from dotenv import load_dotenv
from langsmith import traceable

from ..chunking import chunk_documents
from ..config import RESET_VECTOR_STORE
from ..embeddings import EmbeddingManager
from ..generation import RagGenerator, get_llm
from ..ingestion import load_documents
from ..retrieval import RAGRetriever
from ..vector_store import VectorStore

load_dotenv()


class RagPipeline:
    def __init__(self):
        self._embedding_manager = None
        self._vector_store = None
        self._retriever = None
        self._generator = None

    # ------------------------------------------------------------------
    # Lazy component access (singletons)
    # ------------------------------------------------------------------
    @property
    def embedding_manager(self) -> EmbeddingManager:
        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager()
        return self._embedding_manager

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def retriever(self) -> RAGRetriever:
        if self._retriever is None:
            self._retriever = RAGRetriever(self.vector_store, self.embedding_manager)
        return self._retriever

    @property
    def generator(self) -> RagGenerator:
        if self._generator is None:
            self._generator = RagGenerator(self.retriever, get_llm())
        return self._generator

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    @traceable(name="Build_Index", run_type="chain")
    def build_index(self, reset: bool | None = None) -> int:
        """Load documents, chunk, embed and upsert into the vector store.

        Returns the number of chunks indexed.
        """
        reset = RESET_VECTOR_STORE if reset is None else reset

        print("--- [1/4] Loading Documents ---")
        raw_docs = load_documents()

        print("--- [2/4] Chunking (Agentic) + Embedding ---")
        chunks = chunk_documents(raw_docs, embedding_manager=self.embedding_manager)
        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)

        print("--- [3/4] Preparing Vector Store ---")
        if reset:
            print("Resetting vector store...")
            self.vector_store.reset()

        print("--- [4/4] Inserting Documents & Embeddings ---")
        self.vector_store.add_documents(chunks, embeddings)

        print("--- [5/5] Rebuilding keyword index ---")
        self.retriever.refresh_corpus()

        print(f"Indexing complete. {len(chunks)} chunks indexed.")
        return len(chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    @traceable(name="Answer_Query_Workflow", run_type="chain")
    def answer_query(self, query: str, top_k: int = 5) -> str:
        """Single-shot Q&A (no chat history)."""
        return self.generator.answer_query(query, top_k=top_k)

    def chat(
        self,
        query: str,
        history: List[Dict[str, str]],
        top_k: int = 5,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """One chat turn with session-scoped history. Returns
        (answer, retrieved_docs, answer_type)."""
        return self.generator.answer_chat(query, history, top_k=top_k)

    def stream_chat(
        self,
        query: str,
        history: List[Dict[str, str]],
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], Generator[str, None, None], str]:
        """Same as chat() but streams tokens. Returns
        (retrieved_docs, token_stream, answer_type)."""
        return self.generator.stream_chat(query, history, top_k=top_k)

    def index_stats(self) -> Dict[str, Any]:
        """Collection health info for the API /health endpoint."""
        try:
            return {
                "backend": os.getenv("VECTOR_DB", "QDRANT").upper(),
                "collection": self.vector_store.collection_name,
                "vectors": self.vector_store.count(),
                "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
                "embedding_model": os.getenv(
                    "EMBEDDING_MODEL_NAME", "text-embedding-3-large"
                ),
                "hybrid_search": os.getenv("HYBRID_SEARCH", "true"),
                "rerank": os.getenv("RERANK_ENABLED", "true"),
                "chunk_strategy": os.getenv("CHUNK_STRATEGY", "hybrid"),
            }
        except Exception as e:
            return {"backend": "unknown", "error": str(e)}


_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    """Module-level singleton so API workers share one pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline