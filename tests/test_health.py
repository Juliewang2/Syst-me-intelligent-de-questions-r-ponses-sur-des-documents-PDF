"""
tests/test_health.py
----------------------
Basic smoke tests for the health endpoint and app bootstrap. These
tests do not require a real OpenAI API key since they only exercise
endpoints that don't call the LLM.

Run with:
    pytest -v
"""

import uuid

import pytest

from tests.conftest import register


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def logged_in(make_client):
    client = make_client()
    register(client, f"user_{uuid.uuid4().hex[:8]}")
    return client


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data
    assert "openai_configured" in data


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_login_page_loads(client):
    response = client.get("/login")
    assert response.status_code == 200


def test_upload_page_loads(logged_in):
    response = logged_in.get("/upload")
    assert response.status_code == 200


def test_chat_page_loads(logged_in):
    response = logged_in.get("/chat")
    assert response.status_code == 200


def test_list_documents_empty_ok(logged_in):
    response = logged_in.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["documents"] == []
    assert data["total"] == 0


def test_upload_rejects_non_pdf(logged_in):
    response = logged_in.post(
        "/api/upload",
        files={"files": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400


def test_get_nonexistent_document_404(logged_in):
    response = logged_in.get("/api/documents/does-not-exist")
    assert response.status_code == 404


def test_swagger_docs_available(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema_available(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"]
