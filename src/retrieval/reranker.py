"""Cross-encoder reranking.

A cross-encoder scores (query, passage) pairs jointly — much more accurate
than bi-encoder similarity. We use a small local model
(ms-marco-MiniLM-L-6-v2) that runs on CPU and needs no API key.

The model is loaded lazily and cached process-wide. If it cannot be
loaded (no network for first download, etc.) reranking is disabled with a
warning instead of crashing the request.
"""

import threading
from typing import List, Optional, Tuple

from ..config import RERANK_MODEL_NAME

_rerank_lock = threading.Lock()
_reranker_cache = {}


def get_reranker():
    """Process-wide singleton cross-encoder."""
    if RERANK_MODEL_NAME in _reranker_cache:
        return _reranker_cache[RERANK_MODEL_NAME]
    with _rerank_lock:
        if RERANK_MODEL_NAME in _reranker_cache:
            return _reranker_cache[RERANK_MODEL_NAME]
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(RERANK_MODEL_NAME)
            print(f"Cross-encoder loaded: {RERANK_MODEL_NAME}")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not load reranker {RERANK_MODEL_NAME} ({e}). "
                  "Continuing without reranking.")
            _reranker_cache[RERANK_MODEL_NAME] = None
            return None
        _reranker_cache[RERANK_MODEL_NAME] = model
        return model


def rerank(
    query: str,
    candidates: List[dict],
    top_k: int = 5,
    min_score: float = 0.0,
) -> Tuple[List[dict], bool]:
    """Rerank candidate chunks (each with "id" and "content").

    Returns (reranked_candidates, reranked_ok). Scores are normalized to
    [0, 1] (sigmoid of the cross-encoder raw score) and stored under
    "similarity_score"; the pre-rerank score is preserved in
    "dense_score". If reranking is unavailable the input order is kept and
    reranked_ok=False.
    """
    import math

    model = get_reranker()
    if model is None or not candidates:
        return candidates[:top_k], False

    pairs = [(query, c["content"][:2000]) for c in candidates]
    raw_scores = model.predict(pairs)

    for c, raw in zip(candidates, raw_scores):
        c["dense_score"] = c.get("similarity_score", 0.0)
        c["rerank_score"] = 1.0 / (1.0 + math.exp(-float(raw)))
        c["similarity_score"] = c["rerank_score"]

    ranked = sorted(
        candidates, key=lambda c: c["rerank_score"], reverse=True
    )
    if min_score > 0:
        ranked = [c for c in ranked if c["rerank_score"] >= min_score]
    return ranked[:top_k], True