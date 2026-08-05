"""
RETRIEVAL
---------
Given a developer's query, embed it and pull back the most similar
chunks from the vector store.

Fix vs. the original notebook: the method was defined as `retrive`
(typo) but called elsewhere as `retrieve`/`retriver`, which would
raise AttributeError. Standardized on `retrieve` everywhere below.
"""

from typing import Any, Dict, List

from config import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD
from embeddings import EmbeddingManager
from vector_store import VectorStore


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top k: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            retrieved_docs: List[Dict[str, Any]] = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


if __name__ == "__main__":
    vs = VectorStore()
    em = EmbeddingManager()
    retriever = RAGRetriever(vs, em)
    for doc in retriever.retrieve("Could u explain me the http onConnect"):
        print(doc["rank"], doc["similarity_score"], doc["metadata"].get("source"))