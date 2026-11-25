"""
scripts/reindex.py
--------------------
Rebuilds users' vector indexes with the current embedding model - needed
after switching DEPLOYMENT_MODE (or the embedding model), since vectors
from different models are not comparable and services/vector_store.py
refuses to search an index built with another model.

Every ready document of the user is re-read from its stored PDF,
re-chunked with its stored metadata (title, classification, ...) and
embedded again; the result replaces the user's index for the current
model only, so indexes of other models stay and switching back needs no
re-embedding. A restricted document that the current embedding provider
may not receive is skipped with a FAIL line (same rule as at import).

Usage (project root):
    venv\\Scripts\\python.exe scripts\\reindex.py --owner alice
    venv\\Scripts\\python.exe scripts\\reindex.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document as LCDocument  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from config import get_settings  # noqa: E402
from database import db_session, init_db  # noqa: E402
from models import Document, User  # noqa: E402
from services.chunking_service import chunk_document  # noqa: E402
from services.document_meta import META_FIELDS, DocumentMeta  # noqa: E402
from services.ingestion import ensure_ingestion_allowed  # noqa: E402
from services.pdf_loader import extract_pdf_text  # noqa: E402
from services.sensitivity import ClassificationNotAllowed  # noqa: E402
from services.vector_store import embedding_slug, rebuild_index  # noqa: E402

settings = get_settings()


def stored_meta(document: Document) -> Optional[DocumentMeta]:
    """The metadata the document was imported with (None when corpus
    metadata is off, exactly as at import)."""
    if not settings.corpus_metadata_enabled:
        return None
    values = {field: getattr(document, field) for field in META_FIELDS}
    return DocumentMeta.model_validate({k: v for k, v in values.items() if v is not None})


def document_chunks(document: Document) -> Tuple[List[LCDocument], str]:
    """(chunks, status line) for one stored document."""
    name = document.original_filename
    meta = stored_meta(document)
    try:
        ensure_ingestion_allowed(meta)
    except ClassificationNotAllowed as exc:
        return [], f"FAIL  {name}: {exc}"
    pdf = Path(document.file_path)
    if not pdf.exists():
        return [], f"FAIL  {name}: stored PDF missing ({pdf})"
    chunks = chunk_document(extract_pdf_text(pdf), document_id=document.id, filename=name, meta=meta)
    return chunks, f"OK    {name}: {len(chunks)} chunks"


def reindex_user(db: Session, user: User) -> int:
    """Rebuild one user's index; returns the number of failed documents."""
    documents = db.query(Document).filter(Document.owner_id == user.id, Document.status == "ready").all()
    all_chunks: List[LCDocument] = []
    failures = 0
    for document in documents:
        chunks, line = document_chunks(document)
        print(f"  {line}", flush=True)
        failures += line.startswith("FAIL")
        all_chunks.extend(chunks)
        if chunks:
            document.chunk_count = len(chunks)
    count = rebuild_index(user.id, all_chunks)
    db.commit()
    print(f"{user.username}: {count} chunks indexed with {embedding_slug()}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild vector indexes with the current embedding model.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--owner", help="Username whose index to rebuild")
    target.add_argument("--all", action="store_true", help="Rebuild every user's index")
    args = parser.parse_args()

    init_db()
    failures = 0
    with db_session() as db:
        query = db.query(User)
        users = query.all() if args.all else query.filter(User.username == args.owner).all()
        if not users:
            sys.exit(f"No user named {args.owner!r}.")
        for user in users:
            failures += reindex_user(db, user)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
