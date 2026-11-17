"""
tests/test_corpus_meta.py
---------------------------
Document metadata: validation, the DR_SIMULE safe default, sidecar
loading, and copying onto every chunk.
"""

import json

import pytest
from pydantic import ValidationError

from services.chunking_service import chunk_document
from services.document_meta import DocumentMeta, load_sidecar
from services.pdf_loader import ExtractedPage, ExtractedPDF

_BASE_KEYS = {"document_id", "document_name", "page", "page_end", "chunk_index"}


def _extracted(pages: int = 3) -> ExtractedPDF:
    return ExtractedPDF(
        pages=[ExtractedPage(page_number=i, text=f"Paragraphe {i}. " * 120) for i in range(1, pages + 1)],
        page_count=pages,
    )


def test_metadata_is_copied_onto_every_chunk():
    meta = DocumentMeta(
        doc_id="a-res-77-276-fr",
        title="Résolution 77/276",
        doc_type="resolution",
        source="public",
        classification="PUBLIC",
        language="fr",
        date="2023-03-29",
        un_symbol="A/RES/77/276",
    )
    chunks = chunk_document(_extracted(), document_id="d1", filename="res.pdf", meta=meta)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["doc_id"] == "a-res-77-276-fr"
        assert chunk.metadata["classification"] == "PUBLIC"
        assert chunk.metadata["date"] == "2023-03-29"  # ISO string, for range filters
        assert chunk.metadata["un_symbol"] == "A/RES/77/276"
        assert chunk.metadata["document_id"] == "d1"


def test_without_meta_chunks_are_unchanged():
    chunks = chunk_document(_extracted(), document_id="d1", filename="res.pdf")
    assert all(set(c.metadata) == _BASE_KEYS for c in chunks)


def test_unset_fields_are_not_stored_on_chunks():
    chunks = chunk_document(_extracted(1), document_id="d1", filename="note.pdf", meta=DocumentMeta())
    assert set(chunks[0].metadata) == _BASE_KEYS | {"classification"}


def test_classification_defaults_to_dr_simule():
    assert DocumentMeta().classification == "DR_SIMULE"
    assert DocumentMeta(title="Note", source="interne").classification == "DR_SIMULE"


def test_missing_sidecar_means_dr_simule(tmp_path):
    pdf = tmp_path / "note.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert load_sidecar(pdf).classification == "DR_SIMULE"


def test_sidecar_without_classification_means_dr_simule(tmp_path):
    pdf = tmp_path / "td.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    (tmp_path / "td.json").write_text(json.dumps({"doc_id": "td-1", "version": 2}), encoding="utf-8")

    meta = load_sidecar(pdf)
    assert (meta.doc_id, meta.version, meta.classification) == ("td-1", 2, "DR_SIMULE")


@pytest.mark.parametrize(
    "bad",
    [{"source": "secret"}, {"classification": "TOP_SECRET"}, {"date": "29/03/2023"}, {"version": "two"}],
)
def test_invalid_metadata_is_rejected(bad):
    with pytest.raises(ValidationError):
        DocumentMeta.model_validate(bad)
