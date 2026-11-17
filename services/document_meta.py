"""
services/document_meta.py
---------------------------
Descriptive metadata for a corpus document (title, type, origin,
classification, date, UN symbol...), validated with Pydantic.

Metadata travels with a PDF as a same-named JSON "sidecar"
(`A_RES_77_276-FR.pdf` + `A_RES_77_276-FR.json`), or as a JSON field on
upload. It is copied onto every chunk, so retrieval can filter on it and
answers can cite a document by title and symbol rather than file name.

Classification defaults to DR_SIMULE when unknown. That is the safe side:
a document wrongly treated as restricted only gets more careful handling,
while one wrongly treated as public could be sent somewhere it must not
go. A public document therefore has to be marked PUBLIC explicitly.

All of this is only used when CORPUS_METADATA_ENABLED is on; with it off,
documents and chunks look exactly as before.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

Source = Literal["public", "interne"]
Classification = Literal["PUBLIC", "DR_SIMULE"]

# Metadata fields stored on the Document row and on every chunk.
META_FIELDS = (
    "doc_id",
    "title",
    "doc_type",
    "source",
    "classification",
    "language",
    "date",
    "version",
    "supersedes",
    "un_symbol",
)


class DocumentMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doc_id: Optional[str] = None
    title: Optional[str] = None
    doc_type: Optional[str] = None
    source: Optional[Source] = None
    classification: Classification = "DR_SIMULE"
    language: Optional[str] = None
    date: Optional[datetime.date] = None
    version: Optional[int] = None
    supersedes: Optional[str] = None
    un_symbol: Optional[str] = None

    def chunk_metadata(self) -> dict:
        """Fields to copy onto each chunk. The date is an ISO string so
        date-range filters can compare it as text; unset fields are left
        out rather than stored as None."""
        data = self.model_dump(include=set(META_FIELDS))
        if self.date is not None:
            data["date"] = self.date.isoformat()
        return {k: v for k, v in data.items() if v is not None}


def sidecar_path(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(".json")


def load_sidecar(pdf_path: Path) -> DocumentMeta:
    """The PDF's sidecar metadata, or defaults (DR_SIMULE) if it has none.
    An invalid sidecar raises (pydantic.ValidationError / ValueError)
    rather than being silently replaced by defaults."""
    path = sidecar_path(pdf_path)
    if not path.exists():
        return DocumentMeta()
    return DocumentMeta.model_validate(json.loads(path.read_text(encoding="utf-8")))
