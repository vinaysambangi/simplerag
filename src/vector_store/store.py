"""Vector store wrapper supporting Qdrant and Chroma backends.

The public API is backend-agnostic. `query()` returns a dict shaped like
Chroma's `collection.query()` result so higher-level code stays unchanged.
"""

import os
import uuid
from typing import Any, List

import numpy as np

from ..config import COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL, VECTOR_DB, VECTOR_DIR

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qtypes
except ImportError:  # pragma: no cover - only if qdrant-client is absent
    QdrantClient = None
    qtypes = None

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None


class VectorStore:
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = str(VECTOR_DIR),
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._qdrant = None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)

        if VECTOR_DB == "QDRANT":
            if QdrantClient is None:
                raise ImportError(
                    "qdrant-client is not installed; install it to use the Qdrant backend"
                )
            self._qdrant = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY or None,
                timeout=60.0,
                check_compatibility=False,
            )
            print(f"Qdrant client initialized. Collection: {self.collection_name}")
        else:
            if chromadb is None:
                raise ImportError("chromadb is not installed")
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "API documentation embeddings for RAG"},
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

    # ------------------------------------------------------------------
    # Qdrant helpers
    # ------------------------------------------------------------------
    def _collection_exists(self) -> bool:
        if self._qdrant is None:
            return False
        try:
            return self._qdrant.collection_exists(
                collection_name=self.collection_name
            )
        except Exception:
            return False

    def _ensure_qdrant_collection(self, dim: int) -> None:
        if self._qdrant is None:
            raise RuntimeError("Qdrant client not initialized")
        if self._collection_exists():
            return
        self._qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=qtypes.VectorParams(
                size=dim, distance=qtypes.Distance.COSINE
            ),
            metadata={"description": "API documentation embeddings for RAG"},
        )

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        document_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            document_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        if VECTOR_DB == "QDRANT":
            self._add_to_qdrant(ids, metadatas, document_text, embeddings_list)
        else:
            self._add_to_chroma(ids, metadatas, document_text, embeddings_list)

    def _add_to_qdrant(self, ids, metadatas, document_text, embeddings_list):
        if self._qdrant is None:
            raise RuntimeError("Qdrant client not initialized")
        dim = len(embeddings_list[0]) if embeddings_list else 0
        self._ensure_qdrant_collection(dim)

        batch_size = 256
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points = []
            for i in range(start, end):
                payload = dict(metadatas[i])
                payload["document"] = document_text[i]
                points.append(
                    qtypes.PointStruct(
                        id=ids[i],
                        vector=embeddings_list[i],
                        payload=payload,
                    )
                )
            self._qdrant.upsert(
                collection_name=self.collection_name, points=points
            )
            print(f"  Added batch {start}-{end} of {total}")

        print(f"Successfully added {len(ids)} documents to Qdrant collection")

    def _add_to_chroma(self, ids, metadatas, document_text, embeddings_list):
        if self.collection is None:
            raise RuntimeError("Chroma collection not initialized")
        try:
            batch_size = self.client.get_max_batch_size()
        except Exception:
            batch_size = 4000

        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings_list[start:end],
                metadatas=metadatas[start:end],
                documents=document_text[start:end],
            )
            print(f"  Added batch {start}-{end} of {total}")

        print(f"Successfully added {len(ids)} documents to vector store")
        print(f"Total documents in collection: {self.collection.count()}")

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------
    def query(self, query_embeddings: List[List[float]], n_results: int = 5) -> dict:
        """Backend-agnostic query. Returns Chroma-shaped result dict."""
        if VECTOR_DB == "QDRANT":
            return self._query_qdrant(query_embeddings, n_results)
        if self.collection is None:
            raise RuntimeError("Chroma collection not initialized")
        return self.collection.query(
            query_embeddings=query_embeddings, n_results=n_results
        )

    def _query_qdrant(self, query_embeddings: List[List[float]], n_results: int) -> dict:
        results = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        if self._qdrant is None:
            raise RuntimeError("Qdrant client not initialized")
        if not query_embeddings:
            return results

        try:
            response = self._qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embeddings[0],
                limit=n_results,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            print(f"Error querying Qdrant: {e}")
            return results

        hits = []
        if hasattr(response, "points"):
            hits = response.points or []
        elif hasattr(response, "result"):
            hits = response.result or []

        ids, documents, metadatas, distances = [], [], [], []
        for h in hits:
            ids.append(str(getattr(h, "id", "")))
            payload = getattr(h, "payload", {}) or {}
            documents.append(payload.get("document", ""))
            meta = dict(payload)
            meta.pop("document", None)
            metadatas.append(meta)
            score = getattr(h, "score", None)
            distances.append(1.0 - float(score) if score is not None else 1.0)

        results["ids"].append(ids)
        results["documents"].append(documents)
        results["metadatas"].append(metadatas)
        results["distances"].append(distances)
        return results

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def all_documents(self) -> List[dict]:
        """Return every stored chunk as {"id", "content", "metadata"}.

        Used to build the BM25 keyword index for hybrid retrieval. Scrolls
        the whole collection in pages.
        """
        if VECTOR_DB == "QDRANT":
            return self._all_qdrant()
        if self.collection is None:
            raise RuntimeError("Chroma collection not initialized")
        data = self.collection.get(include=["documents", "metadatas"])
        docs = []
        ids = data.get("ids", []) or []
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []
        for i, doc_id in enumerate(ids):
            docs.append(
                {
                    "id": doc_id,
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            )
        return docs

    def _all_qdrant(self) -> List[dict]:
        if self._qdrant is None:
            raise RuntimeError("Qdrant client not initialized")
        docs = []
        offset = None
        while True:
            response = self._qdrant.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points = response.points or []
            for p in points:
                payload = getattr(p, "payload", {}) or {}
                docs.append(
                    {
                        "id": str(getattr(p, "id", "")),
                        "content": payload.get("document", ""),
                        "metadata": {
                            k: v for k, v in payload.items() if k != "document"
                        },
                    }
                )
            if not points or response.next_page_offset is None:
                break
            offset = response.next_page_offset
        return docs

    def reset(self) -> None:
        """Drop and recreate the collection to remove stale documents."""
        if VECTOR_DB == "QDRANT":
            if self._qdrant is None:
                raise RuntimeError("Qdrant client not initialized")
            try:
                self._qdrant.delete_collection(
                    collection_name=self.collection_name
                )
                print(f"Deleted Qdrant collection: {self.collection_name}")
            except Exception as e:
                print(f"Warning: could not delete Qdrant collection ({e})")
        else:
            if self.client is None:
                raise RuntimeError("Chroma client not initialized")
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
            except Exception:
                import shutil

                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory)
                    print(
                        f"Removed persistent directory: {self.persist_directory}"
                    )

        self._initialize_store()
        print("Vector store reset complete.")

    def count(self) -> int:
        """Number of vectors currently stored."""
        if VECTOR_DB == "QDRANT":
            if self._qdrant is None:
                return 0
            try:
                info = self._qdrant.get_collection(
                    collection_name=self.collection_name
                )
                return info.points_count or 0
            except Exception:
                return 0
        if self.collection is not None:
            return self.collection.count()
        return 0