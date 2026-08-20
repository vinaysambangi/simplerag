"""API route definitions."""

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import CHAT_DB_PATH
from ..core.pipeline import get_pipeline
from ..core.sessions import ChatSessionManager

router = APIRouter()

sessions = ChatSessionManager(CHAT_DB_PATH)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New chat"


class RenameSessionRequest(BaseModel):
    title: str


class SendMessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    sources: List[Dict[str, Any]]
    created_at: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health")
def health():
    stats = get_pipeline().index_stats()
    return {
        "status": "ok" if not stats.get("error") else "degraded",
        "index": stats,
        "sessions": len(sessions.list_sessions()),
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@router.get("/sessions")
def list_sessions():
    return sessions.list_sessions()


@router.post("/sessions")
def create_session(req: Optional[CreateSessionRequest] = None):
    """New chat — starts with an empty history, no context from older chats."""
    return sessions.create_session(title=(req.title if req else None) or "New chat")


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if not sessions.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.patch("/sessions/{session_id}")
def rename_session(session_id: str, req: RenameSessionRequest):
    session = sessions.rename_session(session_id, req.title)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
@router.get("/sessions/{session_id}/messages")
def list_messages(session_id: str):
    if sessions.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions.list_messages(session_id)


@router.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, req: SendMessageRequest):
    """Send a message in a session. The answer uses only this session's
    history as chat context (never other sessions)."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Message content is empty")
    if sessions.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    pipeline = get_pipeline()

    # Capture history BEFORE persisting the current user message, so the
    # message isn't duplicated (build_chat_messages re-adds it augmented
    # with the retrieved context).
    history = sessions.history_for_chat(session_id)

    # Persist user message first.
    user_msg = sessions.add_message(session_id, "user", req.content.strip())
    _auto_title(session_id)

    answer, sources, answer_type = pipeline.chat(req.content.strip(), history)

    assistant_msg = sessions.add_message(
        session_id, "assistant", answer, sources=sources, answer_type=answer_type
    )
    return {"user_message": user_msg, "assistant_message": assistant_msg}


@router.post("/sessions/{session_id}/messages/stream")
async def send_message_stream(session_id: str, req: SendMessageRequest):
    """Streaming variant: yields SSE events.

    Event flow per turn:
      event: sources   (JSON payload of retrieved chunks)
      event: message   (text deltas of the answer)
      event: done      (final persisted assistant message)
    """
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Message content is empty")
    if sessions.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    pipeline = get_pipeline()
    history = sessions.history_for_chat(session_id)
    user_msg = sessions.add_message(session_id, "user", req.content.strip())
    _auto_title(session_id)

    sources, token_stream, answer_type = pipeline.stream_chat(
        req.content.strip(), history
    )

    async def event_stream():
        yield _sse("sources", {"sources": sources, "answer_type": answer_type})

        parts = []
        for token in token_stream:
            parts.append(token)
            yield _sse("message", {"delta": token})

        answer = "".join(parts)
        assistant_msg = sessions.add_message(
            session_id,
            "assistant",
            answer,
            sources=sources,
            answer_type=answer_type,
        )
        yield _sse("done", {"message": assistant_msg})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _auto_title(session_id: str, max_len: int = 40) -> None:
    """Give the session a readable title from its first user message."""
    session = sessions.get_session(session_id)
    if session is None or session["title"] != "New chat":
        return
    messages = sessions.list_messages(session_id)
    if not messages:
        return
    first_user = next((m for m in messages if m["role"] == "user"), None)
    if first_user is None:
        return
    title = " ".join(first_user["content"].split())[:max_len]
    sessions.rename_session(session_id, title or "New chat")