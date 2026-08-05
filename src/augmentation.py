"""
AUGMENTATION
------------
This is the "A" in RAG: takes the retrieved chunks + the developer's
query and assembles the messages that get sent to the LLM.

Split into two pieces on purpose:
  - SYSTEM_PROMPT: static instructions, sent once as the system role.
  - build_user_message(): dynamic per-query content (context + query),
    sent as the user role.

Keeping instructions in `system` and data in `user` is generally more
reliable than folding both into one giant user message (as the
original notebook did), and it means you're not re-sending the whole
instruction block as "user content" every call.
"""

from typing import Any, Dict, List

SYSTEM_PROMPT = """You are a technical assistant helping RTI driver developers understand OEM/vendor API documentation for the device or system they are currently integrating.

You will be given:
1. A developer's query
2. A set of retrieved documentation chunks (fetched via similarity search from a vector database of previously indexed API documentation)

RULES:
1. Answer ONLY using information contained in the provided context chunks below. Do not use prior knowledge of this API, this vendor, or similar APIs to fill gaps.
2. If the retrieved context does not contain enough information to answer the query, say so explicitly — do not guess, infer undocumented behavior, or hallucinate endpoints, parameters, request/response formats, error codes, or command syntax.
3. When you answer, cite which chunk(s) the information came from (e.g. "[Chunk 2]") so the developer can trace it back to the source document.
4. If multiple chunks contain conflicting or overlapping information (e.g. different versions of the same endpoint), point out the conflict rather than silently picking one.
5. Preserve exact technical details verbatim where precision matters — request/response JSON structures, command strings, hex/byte payloads, status/error codes, units, parameter names, and casing. Do not paraphrase or "clean up" these details in a way that changes them.
6. If the query is about implementation (e.g. "how do I structure the driver command for X"), answer at the level of what the documentation specifies (endpoints, payload structure, auth, polling/subscription model) — do not invent driver code unless the retrieved context includes actual reference driver code.
7. Keep responses concise and structured (use headers/bullets/code blocks) — developers are using this to quickly look up integration details, not read prose."""


def format_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Turn retrieved chunk dicts into a numbered context block so the
    model can cite "[Chunk N]" and you can trace it back to a source.
    """
    if not retrieved_docs:
        return ""

    blocks = []
    for doc in retrieved_docs:
        source = doc["metadata"].get("source", "unknown source")
        blocks.append(
            f"[Chunk {doc['rank']}] (source: {source}, "
            f"similarity: {doc['similarity_score']:.3f})\n{doc['content']}"
        )
    return "\n\n".join(blocks)


def build_user_message(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    context = format_context(retrieved_docs)
    if not context:
        return None  # signal "nothing to answer from"

    return f"""Context chunks:
{context}

Developer query:
{query}"""