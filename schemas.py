"""
schemas.py
----------
Pydantic v2 schemas used for API request validation and response
serialization. Kept separate from the SQLAlchemy models (models.py)
to keep the persistence layer and the API contract independently
evolvable.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str = Field(description="Original filename as uploaded by the user")
    file_size_bytes: int
    page_count: int
    chunk_count: int
    status: str
    error_message: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class DocumentSummaryRequest(BaseModel):
    max_key_points: int = Field(default=5, ge=1, le=15)


class DocumentSummaryResponse(BaseModel):
    """Structured output returned by the OpenAI Responses API summary
    generation feature. Mirrors the JSON schema sent to the model."""

    title: str
    overview: str
    key_points: List[str]
    document_type: str
    estimated_reading_time_minutes: int


# ---------------------------------------------------------------------------
# Chat / Conversations
# ---------------------------------------------------------------------------

class SourceChunk(BaseModel):
    document_id: str
    document_name: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    text_snippet: str


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    sources: List[SourceChunk] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: Optional[str] = None
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    message_count: int = 0


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]
    total: int


class ConversationCreateRequest(BaseModel):
    document_id: Optional[str] = Field(
        default=None, description="Restrict retrieval to a single document. Omit to search all documents."
    )
    title: Optional[str] = Field(default=None, max_length=255)


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(
        default=None, description="Existing conversation id. A new one is created if omitted."
    )
    document_id: Optional[str] = Field(
        default=None, description="Restrict retrieval to a single document."
    )
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    conversation_id: str
    message: ChatMessageResponse
    sources: List[SourceChunk] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    openai_configured: bool
    timestamp: datetime
