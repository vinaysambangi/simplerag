"""BM25 keyword index over the stored chunks.

Built once per process from the vector store corpus (see
VectorStore.all_documents()) and cached; invalidated by
refresh_corpus() after every index build.

Used for hybrid retrieval: dense vector search catches semantics, BM25
catches exact tokens — endpoint names, error codes, command strings,
hex/byte payloads — that embeddings alone can miss.
"""

import re
import threading
from typing import Dict, List

from ..config import RRF_K

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - rank_bm25 is a requirements.txt dep
    BM25Okapi = None

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "this", "that", "these",
    "those", "it", "its", "as", "at", "by", "from", "into", "over", "under",
    "will", "can", "should", "may", "must", "not", "no", "if", "then",
    "than", "so", "but", "which", "what", "when", "where", "how", "who",
}

TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-\.]{2,}")

_bm25_lock = threading.Lock()


def tokenize(text: str) -> List[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    def __init__(self, rrf_k: int = RRF_K):
        self.rrf_k = rrf_k
        self._documents: List[dict] = []
        self._bm25: BM25Okapi | None = None
        self._version = 0

    def refresh(self, documents: List[dict]) -> None:
        """Replace the corpus (call after building the index)."""
        with _bm25_lock:
            self._documents = documents
            self._version += 1
            if BM25Okapi is None or not documents:
                self._bm25 = None
                return
            tokenized = [tokenize(d["content"]) for d in documents]
            self._bm25 = BM25Okapi(tokenized)
            print(
                f"BM25 index rebuilt: {len(documents)} documents "
                f"(version {self._version})"
            )

    @property
    def version(self) -> int:
        return self._version

    def search(self, query: str, top_k: int = 50) -> List[dict]:
        """Return the top_k best BM25 matches:
        [{"index": i, "score": float, "doc": {"id", "content", "metadata"}}]
        """
        if self._bm25 is None or not query.strip():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        try:
            scores = self._bm25.get_scores(tokens)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: BM25 scoring failed ({e})")
            return []
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for idx in ranked[:top_k]:
            if scores[idx] > 0:
                results.append(
                    {"index": idx, "score": float(scores[idx]), "doc": self._documents[idx]}
                )
        return results


def rrf_fuse(
    dense_ranked: List[str],
    bm25_ranked: List[str],
    k: int = RRF_K,
) -> List[str]:
    """Reciprocal Rank Fusion: merge two ranked id lists into one.

    Each list is in rank order (best first). Returns fused ids, best first.
    """
    scores: Dict[str, float] = {}
    for ranked in (dense_ranked, bm25_ranked):
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)