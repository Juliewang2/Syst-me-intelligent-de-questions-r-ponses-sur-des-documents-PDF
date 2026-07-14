"""
routers/upload.py
-------------------
Handles PDF upload and ingestion: validates the file, extracts text,
chunks it, generates embeddings, stores vectors in FAISS, and
persists document metadata to the database.

Supports uploading multiple PDFs in a single request.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from config import get_settings
from database import get_db
from models import Document
from schemas import DocumentListResponse, DocumentResponse
from services.chunking_service import chunk_document
from services.pdf_loader import PDFExtractionError, extract_pdf_text, validate_pdf_header
from services.vector_store import add_document_chunks, delete_document as delete_from_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/upload", tags=["Upload"])
settings = get_settings()


def _validate_upload(file: UploadFile) -> None:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{file.filename}' is not a PDF file. Only .pdf files are supported.",
        )


def _save_upload_to_disk(file: UploadFile, document_id: str) -> Path:
    safe_suffix = Path(file.filename).suffix or ".pdf"
    stored_filename = f"{document_id}{safe_suffix}"
    destination = settings.upload_dir / stored_filename

    size_bytes = 0
    with open(destination, "wb") as out_file:
        while chunk := file.file.read(1024 * 1024):
            size_bytes += len(chunk)
            if size_bytes > settings.max_upload_size_bytes:
                out_file.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"'{file.filename}' exceeds the maximum upload size of "
                        f"{settings.max_upload_size_mb} MB."
                    ),
                )
            out_file.write(chunk)

    return destination


def _process_single_pdf(file: UploadFile, db: Session) -> Document:
    _validate_upload(file)

    document_id = str(uuid.uuid4())
    stored_path = _save_upload_to_disk(file, document_id)

    if not validate_pdf_header(stored_path):
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{file.filename}' does not look like a valid PDF file.",
        )

    document = Document(
        id=document_id,
        filename=stored_path.name,
        original_filename=file.filename,
        file_path=str(stored_path),
        file_size_bytes=stored_path.stat().st_size,
        status="processing",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        extracted = extract_pdf_text(stored_path)
        chunks = chunk_document(extracted, document_id=document.id, filename=file.filename)

        if not chunks:
            raise PDFExtractionError(
                "No text chunks could be generated. The PDF may be empty, scanned, or "
                "contain only images without a text layer."
            )

        add_document_chunks(chunks)

        document.page_count = extracted.page_count
        document.chunk_count = len(chunks)
        document.status = "ready"
        document.error_message = None

    except PDFExtractionError as exc:
        logger.warning("PDF extraction failed for %s: %s", file.filename, exc)
        document.status = "failed"
        document.error_message = str(exc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error processing %s", file.filename)
        document.status = "failed"
        document.error_message = f"Unexpected error: {exc}"

    db.commit()
    db.refresh(document)
    return document


@router.post(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload one or more PDF documents",
)
def upload_pdfs(
    files: List[UploadFile] = File(..., description="One or more PDF files to ingest"),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """
    Upload one or more PDF files. Each file is:
      1. Validated (extension, magic bytes, size limit)
      2. Saved to disk under `data/uploads/`
      3. Text-extracted (pypdf / PyPDF2, via LangChain's PyPDFLoader)
      4. Chunked (RecursiveCharacterTextSplitter)
      5. Embedded and stored in the shared FAISS vector store
      6. Recorded in the database with a `ready` or `failed` status
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    results: List[Document] = []
    for file in files:
        try:
            document = _process_single_pdf(file, db)
            results.append(document)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process upload %s", file.filename)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process '{file.filename}': {exc}",
            ) from exc

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d.to_dict()) for d in results],
        total=len(results),
    )
