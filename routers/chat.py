"""
routers/chat.py
-----------------
Chat endpoints that drive the RAG conversation experience:

- POST /api/chat            -> single request/response answer
- POST /api/chat/stream     -> Server-Sent Events streaming answer
- /api/conversations/*      -> conversation history management
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
from models import Conversation
from schemas import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationResponse,
)
from services.conversation_memory import (
    delete_conversation as delete_conversation_service,
    get_or_create_conversation,
    list_conversations,
    load_chat_history,
    save_message,
)
from services.rag_service import generate_answer, generate_answer_stream

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List conversations",
)
def get_conversations(
    document_id: Optional[str] = None, db: Session = Depends(get_db)
) -> ConversationListResponse:
    conversations = list_conversations(db, document_id=document_id)
    return ConversationListResponse(
        conversations=[ConversationResponse.model_validate(c.to_dict()) for c in conversations],
        total=len(conversations),
    )


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
def create_conversation(
    payload: ConversationCreateRequest, db: Session = Depends(get_db)
) -> ConversationResponse:
    conversation = get_or_create_conversation(
        db, conversation_id=None, document_id=payload.document_id, title=payload.title
    )
    return ConversationResponse.model_validate(conversation.to_dict())


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=List[ChatMessageResponse],
    summary="Get all messages in a conversation",
)
def get_conversation_messages(
    conversation_id: str, db: Session = Depends(get_db)
) -> List[ChatMessageResponse]:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return [ChatMessageResponse.model_validate(m.to_dict()) for m in conversation.messages]


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete a conversation and its messages",
)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> dict:
    deleted = delete_conversation_service(db, conversation_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return {"deleted": True, "conversation_id": conversation_id}


# ---------------------------------------------------------------------------
# Chat (non-streaming)
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse, summary="Ask a question (non-streaming)")
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    conversation = get_or_create_conversation(
        db, conversation_id=payload.conversation_id, document_id=payload.document_id
    )

    history = load_chat_history(db, conversation.id)
    save_message(db, conversation.id, role="user", content=payload.message)

    try:
        answer_text, sources = generate_answer(
            payload.message, history, document_id=payload.document_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to generate an answer: {exc}",
        ) from exc

    assistant_message = save_message(
        db, conversation.id, role="assistant", content=answer_text, sources=sources
    )

    return ChatResponse(
        conversation_id=conversation.id,
        message=ChatMessageResponse.model_validate(assistant_message.to_dict()),
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Chat (streaming via Server-Sent Events)
# ---------------------------------------------------------------------------

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("/chat/stream", summary="Ask a question (streamed via Server-Sent Events)")
async def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    conversation = get_or_create_conversation(
        db, conversation_id=payload.conversation_id, document_id=payload.document_id
    )
    conversation_id = conversation.id

    history = load_chat_history(db, conversation_id)
    save_message(db, conversation_id, role="user", content=payload.message)

    async def event_generator() -> AsyncIterator[str]:
        yield _sse_event({"type": "start", "conversation_id": conversation_id})

        full_answer = ""
        final_sources: list = []
        try:
            async for event in generate_answer_stream(
                payload.message, history, document_id=payload.document_id
            ):
                if event["type"] == "sources":
                    final_sources = event["sources"]
                    yield _sse_event({"type": "sources", "sources": final_sources})
                elif event["type"] == "token":
                    full_answer += event["content"]
                    yield _sse_event({"type": "token", "content": event["content"]})
                elif event["type"] == "error":
                    yield _sse_event({"type": "error", "message": event["message"]})
                elif event["type"] == "done":
                    full_answer = event.get("full_answer", full_answer)
                    final_sources = event.get("sources", final_sources)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming chat failed")
            yield _sse_event({"type": "error", "message": str(exc)})
            yield _sse_event({"type": "close"})
            return

        # Persist the assistant's full response with a fresh DB session
        # (the request-scoped session may be closed by the time the
        # generator finishes streaming).
        if full_answer.strip():
            from database import db_session

            with db_session() as fresh_db:
                save_message(
                    fresh_db,
                    conversation_id,
                    role="assistant",
                    content=full_answer,
                    sources=final_sources,
                )

        yield _sse_event({"type": "close"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
