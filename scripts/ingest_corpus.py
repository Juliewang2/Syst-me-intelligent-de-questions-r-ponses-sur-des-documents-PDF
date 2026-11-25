"""
scripts/ingest_corpus.py
--------------------------
Imports a folder of PDFs, each with its same-named .json metadata
sidecar, into one user's library - the batch equivalent of uploading
them in the browser, metadata included.

Each PDF is copied into the upload folder and indexed through
services/ingestion.py, so it is stored exactly like an uploaded file and
appears in that user's document list. A PDF without a sidecar is
imported as DR_SIMULE (the safe default); an invalid sidecar stops that
file with an error instead of being guessed around.

Re-running is safe: a PDF whose doc_id that user already has (status
ready) is skipped, so fixing one sidecar and re-running doesn't duplicate
everything else. Two *versions* of a document must therefore have
different doc_ids (link them with `supersedes`).

A DR_SIMULE file is refused (FAIL line) when the deployment mode's
embedding provider may not receive restricted text - in openai or
mistral mode, import internal documents with DEPLOYMENT_MODE=hybride or
souverain.

Requires CORPUS_METADATA_ENABLED=true (in .env), since without it the
metadata would not be stored.

Usage (project root):
    venv\\Scripts\\python.exe scripts\\ingest_corpus.py --dir data\\corpus\\public --owner alice
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from config import get_settings  # noqa: E402
from database import db_session, init_db  # noqa: E402
from models import Document, User  # noqa: E402
from services.document_meta import DocumentMeta, load_sidecar  # noqa: E402
from services.ingestion import apply_meta, ensure_ingestion_allowed, index_document  # noqa: E402
from services.sensitivity import ClassificationNotAllowed  # noqa: E402

settings = get_settings()


def find_user(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def already_ingested(db: Session, owner_id: str, doc_id: Optional[str]) -> bool:
    if not doc_id:
        return False
    return (
        db.query(Document)
        .filter(Document.owner_id == owner_id, Document.doc_id == doc_id, Document.status == "ready")
        .first()
        is not None
    )


def ingest_pdf(db: Session, owner_id: str, pdf: Path, meta: DocumentMeta) -> Document:
    """Copy the PDF into the upload folder and index it with its metadata."""
    document_id = str(uuid.uuid4())
    stored_path = settings.upload_dir / f"{document_id}.pdf"
    shutil.copyfile(pdf, stored_path)

    document = Document(
        id=document_id,
        owner_id=owner_id,
        filename=stored_path.name,
        original_filename=pdf.name,
        file_path=str(stored_path),
        file_size_bytes=stored_path.stat().st_size,
        status="processing",
    )
    apply_meta(document, meta)
    db.add(document)
    db.commit()
    db.refresh(document)
    return index_document(db, document, stored_path, pdf.name, meta)


def ingest_one(db: Session, owner_id: str, pdf: Path) -> str:
    """Import one PDF; returns a one-line status for the log."""
    try:
        meta = load_sidecar(pdf)
    except (ValidationError, ValueError) as exc:
        return f"FAIL  {pdf.name}: invalid sidecar - {exc}"
    try:
        ensure_ingestion_allowed(meta)
    except ClassificationNotAllowed as exc:
        return f"FAIL  {pdf.name}: {exc}"
    if already_ingested(db, owner_id, meta.doc_id):
        return f"SKIP  {pdf.name}: doc_id {meta.doc_id!r} already imported"
    document = ingest_pdf(db, owner_id, pdf, meta)
    if document.status != "ready":
        return f"FAIL  {pdf.name}: {document.error_message}"
    return f"OK    {pdf.name}: {document.chunk_count} chunks, {meta.classification}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a folder of PDFs (+ .json metadata).")
    parser.add_argument("--dir", required=True, type=Path, help="Folder containing the PDFs")
    parser.add_argument("--owner", required=True, help="Username that will own the documents")
    args = parser.parse_args()

    if not settings.corpus_metadata_enabled:
        sys.exit("Set CORPUS_METADATA_ENABLED=true in .env first: otherwise metadata is not stored.")
    pdfs = sorted(args.dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDF found in {args.dir}")

    init_db()
    failures = 0
    with db_session() as db:
        user = find_user(db, args.owner)
        if user is None:
            sys.exit(f"No user named {args.owner!r}; register it in the app first.")
        for pdf in pdfs:
            line = ingest_one(db, user.id, pdf)
            failures += line.startswith("FAIL")
            print(line, flush=True)

    print(f"\n{len(pdfs)} PDF(s) processed, {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
