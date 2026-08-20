"""Embedding model wrapper.

Provider "openai" (default): OpenAI text-embedding-3-large — best quality
for technical/API documentation, batched with retry + backoff.

Provider "local": sentence-transformers model running on this machine
(free, offline; slower on CPU for large corpora).

Public interface stays the same either way:
    generate_embeddings(texts: List[str]) -> np.ndarray
"""

import os
import time
from typing import List

import numpy as np

from ..config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROVIDER,
)


class EmbeddingManager:
    def __init__(self, provider: str | None = None, model_name: str | None = None):
        self.provider = (provider or EMBEDDING_PROVIDER).lower()
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self._openai_client = None
        self._local_model = None
        self._load()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _load(self):
        if self.provider == "openai":
            self._load_openai()
        else:
            self._load_local()

    def _load_openai(self):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        self._openai_client = OpenAI(api_key=api_key)
        print(f"Embedding provider: openai ({self.model_name})")

    def _load_local(self):
        from sentence_transformers import SentenceTransformer

        print(f"Loading local embedding model: {self.model_name}")
        self._local_model = SentenceTransformer(self.model_name)

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self.provider == "openai":
            return self._generate_openai(texts)
        return self._generate_local(texts)

    def _generate_local(self, texts: List[str]) -> np.ndarray:
        if self._local_model is None:
            raise RuntimeError("Local embedding model not loaded")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self._local_model.encode(
            texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=True
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _generate_openai(self, texts: List[str]) -> np.ndarray:
        if self._openai_client is None:
            raise RuntimeError("OpenAI client not loaded")
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        vectors: List[List[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            vectors.extend(self._embed_batch_with_retry(batch))
        print(f"Generated {len(vectors)} embeddings ({self.model_name})")
        return np.asarray(vectors, dtype=np.float32)

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        last_error: Exception | None = None
        for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
            try:
                response = self._openai_client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                ordered = sorted(
                    (d.index, d.embedding) for d in response.data
                )
                return [emb for _, emb in ordered]
            except Exception as e:  # noqa: BLE001 - retry on any transient error
                last_error = e
                wait = 2 ** attempt
                print(
                    f"Embedding API call failed (attempt {attempt}/{EMBEDDING_MAX_RETRIES}): "
                    f"{e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
        raise RuntimeError(
            f"OpenAI embedding failed after {EMBEDDING_MAX_RETRIES} attempts: {last_error}"
        )