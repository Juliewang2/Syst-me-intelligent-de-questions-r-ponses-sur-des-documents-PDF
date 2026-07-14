"""
tests/test_health.py
----------------------
Basic smoke tests for the health endpoint and app bootstrap. These
tests do not require a real OpenAI API key since they only exercise
endpoints that don't call the LLM.

Run with:
    pytest -v
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data
    assert "openai_configured" in data


def test_index_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_upload_page_loads():
    response = client.get("/upload")
    assert response.status_code == 200


def test_chat_page_loads():
    response = client.get("/chat")
    assert response.status_code == 200


def test_list_documents_empty_ok():
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["total"], int)


def test_upload_rejects_non_pdf():
    response = client.post(
        "/api/upload",
        files={"files": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400


def test_get_nonexistent_document_404():
    response = client.get("/api/documents/does-not-exist")
    assert response.status_code == 404


def test_swagger_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"]
