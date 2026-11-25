"""
tests/test_ingestion_guard.py
-------------------------------
Restricted (DR_SIMULE) documents are refused at import when the
embedding provider may not receive them: upload endpoint (422), the
batch import script (FAIL line) and index_document itself.
"""

import json
from pathlib import Path

import pytest

from tests.conftest import register
from models import Document
from services import sensitivity
from services.document_meta import DocumentMeta
from services.ingestion import index_document
from services.sensitivity import ClassificationNotAllowed, check_ingestion_allowed


@pytest.fixture
def mode(monkeypatch):
    def _set(name: str) -> None:
        monkeypatch.setattr(sensitivity.settings, "deployment_mode", name)
        monkeypatch.setattr(sensitivity.settings, "corpus_metadata_enabled", True)

    return _set


@pytest.mark.parametrize("name, allowed", [("openai", False), ("mistral", False), ("hybride", True), ("souverain", True)])
def test_restricted_documents_need_a_local_embedding_provider(mode, name, allowed):
    mode(name)
    check_ingestion_allowed("PUBLIC")  # always fine
    if allowed:
        check_ingestion_allowed("DR_SIMULE")
    else:
        with pytest.raises(ClassificationNotAllowed, match="DEPLOYMENT_MODE=hybride"):
            check_ingestion_allowed("DR_SIMULE")


def test_upload_refuses_restricted_document_in_openai_mode(mode, make_client):
    mode("openai")
    client = make_client()
    register(client, "uploader")
    pdf = ("note.pdf", b"%PDF-1.4 not really parsed", "application/pdf")

    for metadata in (json.dumps({"classification": "DR_SIMULE"}), None):  # None = DR_SIMULE default
        data = {"metadata": metadata} if metadata else {}
        response = client.post("/api/upload", files={"files": pdf}, data=data)
        assert response.status_code == 422, response.text
        assert "DR_SIMULE" in response.json()["detail"]
    assert client.get("/api/documents").json()["total"] == 0


def test_ingest_script_reports_refusal(mode, tmp_path):
    from scripts.ingest_corpus import ingest_one

    mode("openai")
    pdf = tmp_path / "td-12.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    (tmp_path / "td-12.json").write_text(json.dumps({"doc_id": "td-12", "classification": "DR_SIMULE"}))

    line = ingest_one(db=None, owner_id="someone", pdf=pdf)  # refused before any DB access
    assert line.startswith("FAIL  td-12.pdf") and "DR_SIMULE" in line


def test_index_document_refuses_even_if_caller_did_not_check(mode, tmp_path):
    class FakeSession:
        def commit(self):
            pass

        def refresh(self, obj):
            pass

    mode("openai")
    document = Document(id="d1", owner_id="o", filename="d1.pdf", original_filename="d1.pdf",
                        file_path=str(tmp_path / "d1.pdf"), file_size_bytes=0, status="processing")
    result = index_document(FakeSession(), document, Path(document.file_path), "d1.pdf",
                            DocumentMeta(classification="DR_SIMULE"))
    assert result.status == "failed" and "DR_SIMULE" in result.error_message
