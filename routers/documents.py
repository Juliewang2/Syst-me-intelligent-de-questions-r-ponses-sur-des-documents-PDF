"""
routers/documents.py
----------------------
CRUD-style endpoints for managing previously uploaded documents:
listing, fetching, deleting, and generating a structured AI summary
(demonstrating the OpenAI Responses API's structured output feature).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models import Document, User
from schemas import (
    DocumentListResponse,
    DocumentResponse,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
)
from services.pdf_loader import extract_pdf_text
from services.auth_service import get_current_user
from services.rag_service import generate_structured_summary
from services.vector_store import delete_document as delete_from_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["Documents"])


def get_owned_document(db: Session, user: User, document_id: str) -> Document:
    """The user's document, or 404 (also for other users' documents, so
    their ids can't even be probed)."""
    document = db.get(Document, document_id)
    if document is None or document.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return document


@router.get("", response_model=DocumentListResponse, summary="List all uploaded documents")
def list_documents(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> DocumentListResponse:
    documents = (
        db.query(Document)
        .filter(Document.owner_id == user.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d.to_dict()) for d in documents],
        total=len(documents),
    )


@router.get("/{document_id}", response_model=DocumentResponse, summary="Get a single document")
def get_document(
    document_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> DocumentResponse:
    document = get_owned_document(db, user, document_id)
    return DocumentResponse.model_validate(document.to_dict())


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document and its vector index entries",
)
def delete_document(
    document_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> dict:
    document = get_owned_document(db, user, document_id)

    removed_chunks = delete_from_vector_store(user.id, document_id)

    file_path = Path(document.file_path)
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError as exc:
            logger.warning("Could not remove file %s: %s", file_path, exc)

    db.delete(document)
    db.commit()

    return {
        "deleted": True,
        "document_id": document_id,
        "removed_vector_chunks": removed_chunks,
    }


@router.post(
    "/{document_id}/summary",
    response_model=DocumentSummaryResponse,
    summary="Generate a structured AI summary (OpenAI Responses API + JSON Schema)",
)
def summarize_document(
    document_id: str,
    payload: DocumentSummaryRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentSummaryResponse:
    """
    Demonstrates the raw OpenAI Responses API with structured output:
    the model is constrained to return JSON matching a fixed schema
    (title, overview, key_points, document_type, estimated reading
    time), which we validate and cache on the document record.
    """
    document = get_owned_document(db, user, document_id)
    if document.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document is not ready for summarization (status: {document.status}).",
        )

    file_path = Path(document.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="The original PDF file is no longer available on disk.",
        )

    try:
        extracted = extract_pdf_text(file_path)
        summary_dict = generate_structured_summary(
            extracted.full_text, max_key_points=payload.max_key_points
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Structured summary generation failed for %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to generate summary: {exc}",
        ) from exc

    import json

    document.summary = json.dumps(summary_dict)
    db.commit()

    return DocumentSummaryResponse.model_validate(summary_dict)
