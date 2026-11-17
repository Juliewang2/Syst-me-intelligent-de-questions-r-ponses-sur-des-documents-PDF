"""
services/ingestion.py
-----------------------
Turns a stored PDF into a searchable document: extract text (with OCR),
chunk it, embed the chunks into the owner's index, and record the
outcome on its `Document` row.

Both ways documents arrive - the upload endpoint and the
scripts/ingest_corpus.py batch import - go through `index_document`, so
a corpus file ends up stored exactly like an uploaded one, metadata
included.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from models import Document
from services.chunking_service import chunk_document
from services.document_meta import META_FIELDS, DocumentMeta
from services.pdf_loader import PDFExtractionError, extract_pdf_text
from services.vector_store import add_document_chunks

logger = logging.getLogger(__name__)


def apply_meta(document: Document, meta: Optional[DocumentMeta]) -> None:
    """Copy metadata fields onto the Document row (no-op without meta)."""
    if meta is None:
        return
    for field in META_FIELDS:
        setattr(document, field, getattr(meta, field))


def index_document(
    db: Session,
    document: Document,
    pdf_path: Path,
    display_name: str,
    meta: Optional[DocumentMeta] = None,
) -> Document:
    """Extract, chunk and embed `pdf_path` into the owner's index, then
    mark the document ready or failed. Never raises for a bad PDF: the
    error is stored on the document instead."""
    try:
        extracted = extract_pdf_text(pdf_path)
        chunks = chunk_document(extracted, document_id=document.id, filename=display_name, meta=meta)

        if not chunks:
            raise PDFExtractionError(
                "No text chunks could be generated. The PDF may be empty, scanned, or "
                "contain only images without a text layer."
            )

        add_document_chunks(document.owner_id, chunks)

        if extracted.ocr_page_count:
            logger.info("%s: %d page(s) read with OCR", display_name, extracted.ocr_page_count)

        document.page_count = extracted.page_count
        document.chunk_count = len(chunks)
        document.status = "ready"
        document.error_message = None

    except PDFExtractionError as exc:
        logger.warning("PDF extraction failed for %s: %s", display_name, exc)
        document.status = "failed"
        document.error_message = str(exc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error processing %s", display_name)
        document.status = "failed"
        document.error_message = f"Unexpected error: {exc}"

    db.commit()
    db.refresh(document)
    return document
