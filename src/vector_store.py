"""
VECTOR STORE
------------
Thin wrapper around a persistent Chroma collection: stores chunk
text + embedding + metadata, keyed by a generated id.
"""

import os
import uuid
from typing import Any, List

import numpy as np
import chromadb

from config import VECTOR_DIR, COLLECTION_NAME


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
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "API documentation embeddings for RAG"},
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def _max_batch_size(self) -> int:
        """Ask the running Chroma instance for its actual limit rather
        than hardcoding a number that might drift between versions."""
        try:
            return self.client.get_max_batch_size()
        except Exception:
            return 4000  # conservative fallback if the API isn't available

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        document_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            document_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        batch_size = self._max_batch_size()

        try:
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

            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def reset(self) -> None:
        """Drop and recreate the collection to remove stale/duplicate documents.

        This tries to use the client's collection delete API. If that's not
        available it falls back to removing the persistent directory and
        reinitializing the client.
        """
        try:
            # Prefer a targeted delete API when available
            if hasattr(self.client, "delete_collection"):
                try:
                    self.client.delete_collection(name=self.collection_name)
                    print(f"Deleted collection: {self.collection_name}")
                except TypeError:
                    # some chroma versions use a different signature
                    self.client.delete_collection(self.collection_name)
                    print(f"Deleted collection: {self.collection_name}")
            else:
                # Fallback: remove the persistent directory entirely
                import shutil

                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory)
                    print(f"Removed persistent directory: {self.persist_directory}")

            # Reinitialize the client and collection
            self._initialize_store()
            print("Vector store reset complete.")
        except Exception as e:
            print(f"Error resetting vector store: {e}")
            raise


if __name__ == "__main__":
    vs = VectorStore()
    print(vs.collection.count())