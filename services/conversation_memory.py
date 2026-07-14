"""
services/conversation_memory.py
----------------------------------
Implements conversation memory backed by SQLite (via SQLAlchemy),
rather than an in-process/ephemeral LangChain memory object. This
means chat history survives application restarts and can be listed,
inspected, and resumed from the UI.

The module exposes helpers to:
  - fetch or create a Conversation
  - load recent turns as LangChain `BaseMessage` objects (for prompt
    construction)
  - persist new user/assistant turns
"""

from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.orm import Session

from config import get_settings
from models import ChatMessage, Conversation

settings = get_settings()


def get_or_create_conversation(
    db: Session,
    conversation_id: Optional[str],
    document_id: Optional[str] = None,
    title: Optional[str] = None,
) -> Conversation:
    """Fetch an existing conversation by id, or create a new one."""
    if conversation_id:
        conversation = db.get(Conversation, conversation_id)
        if conversation is not None:
            return conversation

    conversation = Conversation(
        document_id=document_id,
        title=title or "New Conversation",
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def load_chat_history(
    db: Session,
    conversation_id: str,
    max_messages: Optional[int] = None,
) -> List[BaseMessage]:
    """Load the most recent messages of a conversation as LangChain
    message objects, in chronological order, ready to feed into a
    prompt's `chat_history` placeholder."""
    limit = max_messages or settings.max_history_messages

    messages: List[ChatMessage] = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    messages.reverse()

    history: List[BaseMessage] = []
    for msg in messages:
        if msg.role == "user":
            history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            history.append(AIMessage(content=msg.content))
        # system messages are intentionally excluded from the replayed
        # history; the RAG system prompt is injected fresh each turn.
    return history


def save_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    sources: Optional[list] = None,
) -> ChatMessage:
    """Persist a single chat turn."""
    message = ChatMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=json.dumps(sources) if sources else None,
    )
    db.add(message)

    conversation = db.get(Conversation, conversation_id)
    if conversation is not None:
        # Auto-title new conversations from the first user message.
        if conversation.title in (None, "", "New Conversation") and role == "user":
            conversation.title = (content[:60] + "...") if len(content) > 60 else content

    db.commit()
    db.refresh(message)
    return message


def list_conversations(db: Session, document_id: Optional[str] = None) -> List[Conversation]:
    query = db.query(Conversation)
    if document_id:
        query = query.filter(Conversation.document_id == document_id)
    return query.order_by(Conversation.updated_at.desc()).all()


def delete_conversation(db: Session, conversation_id: str) -> bool:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        return False
    db.delete(conversation)
    db.commit()
    return True
