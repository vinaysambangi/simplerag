"""Retrieval: hybrid dense + BM25 search with cross-encoder reranking.

Pipeline per query:
  1. Dense: embed the query, pull top HYBRID_TOP_K from the vector store.
  2. Sparse: BM25 keyword match over the same corpus (catches exact
     endpoint names, error codes, command strings).
  3. Fuse: Reciprocal Rank Fusion merges both rankings.
  4. Filter: optional metadata filter (source, endpoint, ...) on candidates.
  5. Rerank: local cross-encoder re-scores the fused candidates; the top_k
     highest-scoring chunks are returned with normalized scores.

If BM25 or the reranker is unavailable the pipeline degrades gracefully to
plain vector search.
"""

import threading
from typing import Any, Dict, List, Optional

from langsmith import traceable

from ..config import (
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    HYBRID_SEARCH,
    HYBRID_TOP_K,
    METADATA_FILTER,
    RERANK_ENABLED,
    RERANK_MIN_SCORE,
)
from ..embeddings import EmbeddingManager
from ..vector_store import VectorStore
from .bm25 import BM25Index, rrf_fuse
from .reranker import rerank

_corpus_lock = threading.Lock()


class RAGRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.bm25 = BM25Index()
        self._corpus_loaded = False

    # ------------------------------------------------------------------
    # Corpus management (for BM25)
    # ------------------------------------------------------------------
    def refresh_corpus(self) -> None:
        """Reload the full corpus from the vector store after indexing."""
        with _corpus_lock:
            docs = self.vector_store.all_documents()
            self.bm25.refresh(docs)
            self._corpus_loaded = True

    def _ensure_corpus(self) -> None:
        if self._corpus_loaded:
            return
        with _corpus_lock:
            if self._corpus_loaded:
                return
            docs = self.vector_store.all_documents()
            self.bm25.refresh(docs)
            self._corpus_loaded = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    @traceable(name="Hybrid_Retrieve", run_type="retriever")
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_rerank: bool = RERANK_ENABLED,
    ) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")

        dense_ranked = self._dense_search(query, HYBRID_TOP_K)
        if not dense_ranked:
            print("No documents found in vector store")
            return []

        candidates: List[Dict[str, Any]] = []
        if HYBRID_SEARCH:
            self._ensure_corpus()
            bm25_hits = self.bm25.search(query, top_k=HYBRID_TOP_K)
            if bm25_hits:
                by_id = {d["id"]: d for d in dense_ranked}
                fused_ids = rrf_fuse(
                    [d["id"] for d in dense_ranked],
                    [h["doc"]["id"] for h in bm25_hits],
                )
                candidates = [by_id[doc_id] for doc_id in fused_ids if doc_id in by_id]
            else:
                candidates = dense_ranked
        else:
            candidates = dense_ranked

        candidates = self._apply_metadata_filter(
            candidates, metadata_filter or METADATA_FILTER
        )

        if not candidates:
            print("No candidates after filtering — returning unfiltered top-k fallback")
            candidates = self._apply_metadata_filter(dense_ranked, None)[:top_k]
            if not candidates:
                return []

        if use_rerank and len(candidates) > 1:
            ranked, reranked_ok = rerank(
                query, candidates, top_k=top_k, min_score=RERANK_MIN_SCORE
            )
            if reranked_ok:
                ranked = self._apply_score_threshold(ranked, score_threshold)
                if ranked:
                    for i, c in enumerate(ranked):
                        c["rank"] = i + 1
                    return ranked
                print("No candidates passed score threshold after rerank")

        for i, c in enumerate(candidates[:top_k]):
            c["rank"] = i + 1
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _dense_search(self, query: str, n: int) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        try:
            results = self.vector_store.query(
                query_embeddings=[query_embedding.tolist()], n_results=n
            )
        except Exception as e:  # noqa: BLE001
            print(f"Error during dense retrieval: {e}")
            return []

        if not (results.get("documents") and results["documents"][0]):
            return []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: List[Dict[str, Any]] = []
        for i, (doc_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            similarity = min(1.0, max(0.0, 1 - distance))
            retrieved.append(
                {
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "similarity_score": similarity,
                    "distance": distance,
                }
            )
        return retrieved

    @staticmethod
    def _apply_metadata_filter(
        candidates: List[Dict[str, Any]],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not metadata_filter:
            return candidates
        filtered = []
        for c in candidates:
            meta = c.get("metadata", {}) or {}
            match = True
            for key, value in metadata_filter.items():
                actual = meta.get(key)
                if isinstance(value, (list, tuple, set)):
                    if actual not in value:
                        match = False
                        break
                elif actual != value:
                    match = False
                    break
            if match:
                filtered.append(c)
        return filtered

    @staticmethod
    def _apply_score_threshold(
        candidates: List[Dict[str, Any]], threshold: float
    ) -> List[Dict[str, Any]]:
        if threshold <= 0:
            return candidates
        return [c for c in candidates if c.get("similarity_score", 0) >= threshold]