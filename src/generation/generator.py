"""Generation: tie retrieval + augmentation together and call the LLM."""

import os
from typing import Any, Dict, Generator, List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

from ..config import (
    DEFAULT_TOP_K,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_HISTORY_TURNS,
)
from ..retrieval import RAGRetriever
from .prompts import (
    build_chat_messages,
    build_user_message,
    detect_query_type,
    system_prompt_for,
)

load_dotenv()


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
        model=LLM_MODEL,
        max_completion_tokens=LLM_MAX_TOKENS,
    )


class RagGenerator:
    """Answers a single query or a chat turn (with session-scoped history)."""

    def __init__(self, retriever: RAGRetriever, llm: ChatOpenAI):
        self.retriever = retriever
        self.llm = llm

    # ------------------------------------------------------------------
    # Single-shot query (no chat history)
    # ------------------------------------------------------------------
    @traceable(name="RAG_Answer_Query", run_type="chain")
    def answer_query(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        self._debug_retrieved(retrieved_docs)

        user_message = build_user_message(query, retrieved_docs)
        if user_message is None:
            return "No relevant context found to answer the question."

        context_text = "".join(d.get("content", "") for d in retrieved_docs)
        response = self.llm.invoke(
            [
                {"role": "system", "content": system_prompt_for(query, context_text)},
                {"role": "user", "content": user_message},
            ]
        )
        return response.content

    # ------------------------------------------------------------------
    # Chat turn (session-scoped history)
    # ------------------------------------------------------------------
    @traceable(name="RAG_Prepare_Chat", run_type="chain")
    def prepare_chat(
        self,
        query: str,
        history: List[Dict[str, str]],
        top_k: int = DEFAULT_TOP_K,
        max_history_turns: int = MAX_HISTORY_TURNS,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], str]:
        """Retrieve fresh context for the current question and assemble the
        full message list, including recent history from the current
        session only. Returns (messages, retrieved_docs, answer_type)."""
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        self._debug_retrieved(retrieved_docs)

        # history is a list of {"role", "content"}; a "turn" = user+assistant
        trimmed_history = history[-max_history_turns * 2 :]
        messages = build_chat_messages(query, retrieved_docs, trimmed_history)
        context_text = "".join(d.get("content", "") for d in retrieved_docs)
        answer_type = detect_query_type(query, context_text)
        return messages, retrieved_docs, answer_type

    @traceable(name="RAG_Answer_Chat", run_type="chain")
    def answer_chat(
        self,
        query: str,
        history: List[Dict[str, str]],
        top_k: int = DEFAULT_TOP_K,
        max_history_turns: int = MAX_HISTORY_TURNS,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        messages, retrieved_docs, answer_type = self.prepare_chat(
            query, history, top_k, max_history_turns
        )
        response = self.llm.invoke(messages)
        return response.content, retrieved_docs, answer_type

    @traceable(name="RAG_Answer_Chat_Stream", run_type="chain")
    def stream_chat(
        self,
        query: str,
        history: List[Dict[str, str]],
        top_k: int = DEFAULT_TOP_K,
        max_history_turns: int = MAX_HISTORY_TURNS,
    ) -> Tuple[List[Dict[str, Any]], Generator[str, None, None], str]:
        """Same as answer_chat but streams answer text tokens.

        Returns (retrieved_docs, token_generator, answer_type) — sources are
        available immediately, before the first token arrives.
        """
        messages, retrieved_docs, answer_type = self.prepare_chat(
            query, history, top_k, max_history_turns
        )

        def _stream():
            for chunk in self.llm.stream(messages):
                content = getattr(chunk, "content", None)
                if content:
                    yield content

        return retrieved_docs, _stream(), answer_type

    @staticmethod
    def _debug_retrieved(retrieved_docs):
        for d in retrieved_docs:
            source = d["metadata"].get("source", "?")
            preview = d["content"][:100].replace("\n", " ")
            print(f"  [{d['similarity_score']:.3f}] {source} :: {preview}...")