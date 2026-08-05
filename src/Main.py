"""
Orchestrates the full pipeline, in stage order:

  INGESTION -> CHUNKING -> EMBEDDING -> VECTOR STORE -> RETRIEVAL + AUGMENTATION -> GENERATION

Run once to build the store, then call answer_query() as many times
as you like for different developer questions.
"""

from ingestion import load_documents
from chunking import chunk_documents
from embeddings import EmbeddingManager
from vector_store import VectorStore
from retrieval import RAGRetriever
from generation import get_llm, answer_query


def build_index():
    """Run once (or whenever source docs change) to (re)populate the vector store."""
    raw_docs = load_documents()
    chunks = chunk_documents(raw_docs)

    embedding_manager = EmbeddingManager()
    texts = [c.page_content for c in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    vector_store = VectorStore()
    # reset store to avoid stale/duplicate chunks from previous runs
    try:
        vector_store.reset()
    except Exception:
        # if reset fails, continue with existing collection to avoid data loss
        print("Warning: failed to reset vector store; continuing without reset")

    vector_store.add_documents(chunks, embeddings)

    return vector_store, embedding_manager


def main():
    vector_store, embedding_manager = build_index()
    retriever = RAGRetriever(vector_store, embedding_manager)
    llm = get_llm()

    # query = "How grouping is done with the denon device"
    # answer = answer_query(query, retriever, llm)
    # print("\n--- ANSWER ---")
    # print(answer)


if __name__ == "__main__":
    main()