"""
GENERATION (Query -> Response)
-------------------------------
Ties retrieval + augmentation together and calls the LLM.

Fixes vs. the original notebook:
  - `simpleRag` called `rag_retriever.retriver(...)` (typo, and used
    the module-level `rag_retriever` instead of the `retriever` arg
    that was passed in) — now uses the passed-in retriever correctly.
  - System instructions now go in the `system` role via
    augmentation.SYSTEM_PROMPT, instead of being duplicated inside a
    single long user message.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, DEFAULT_TOP_K
from retrieval import RAGRetriever
from augmentation import SYSTEM_PROMPT, build_user_message

load_dotenv()


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
        model=LLM_MODEL,
        max_completion_tokens=LLM_MAX_TOKENS,
    )


def answer_query(
    query: str,
    retriever: RAGRetriever,
    llm: ChatOpenAI,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """
    Full query -> response cycle: retrieve relevant chunks, build the
    augmented prompt, call the LLM, return the answer text.
    """
    retrieved_docs = retriever.retrieve(query, top_k=top_k)

    user_message = build_user_message(query, retrieved_docs)
    if user_message is None:
        return "No relevant context found to answer the question."

    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )
    return response.content


if __name__ == "__main__":
    from embeddings import EmbeddingManager
    from vector_store import VectorStore

    vs = VectorStore()
    em = EmbeddingManager()
    retriever = RAGRetriever(vs, em)
    llm = get_llm()

    query = "How grouping is done with the denon device"
    print(answer_query(query, retriever, llm))