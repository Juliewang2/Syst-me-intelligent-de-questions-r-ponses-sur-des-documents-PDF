"""
tests/conftest.py
-------------------
Shared pytest setup. Runs before any test module is imported, so the
environment variables below are in place before `config.get_settings()`
is first called and cached: tests get a throwaway database and storage
directories, and never touch real data in `data/`.
"""

import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="pdf-chat-tests-"))
os.environ.update(
    {
        "OPENAI_API_KEY": "sk-test-placeholder-key",
        "DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
        "UPLOAD_DIR": str(_TMP / "uploads"),
        "VECTOR_STORE_DIR": str(_TMP / "vector_store"),
        "APP_ENV": "test",
        # The reranker calls the OpenAI API; tests that need it mock it.
        "RERANK_ENABLED": "false",
        "HYBRID_SEARCH_ENABLED": "true",
        "MIN_VECTOR_SIMILARITY": "0.2",
        "REGISTRATION_INVITE_CODE": "",
    }
)

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402


class HashingEmbeddings(Embeddings):
    """Offline stand-in for OpenAI embeddings: a normalized bag-of-words
    vector (each word hashed into one of 4096 buckets). Texts sharing
    words get a high cosine similarity, unrelated texts about zero -
    enough to test retrieval logic without network calls."""

    dim = 4096

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Route all vector store embedding calls to HashingEmbeddings and
    start from empty indexes."""
    from services import keyword_search, vector_store

    def _reset():
        vector_store.reset_cache()
        keyword_search.reset_cache()
        shutil.rmtree(vector_store.settings.vector_store_dir, ignore_errors=True)

    monkeypatch.setattr(vector_store, "get_embeddings", lambda: HashingEmbeddings())
    _reset()
    yield
    _reset()


@pytest.fixture
def make_client():
    """Factory for TestClients; each one has its own cookie jar, i.e.
    acts as a separate browser / user. The context manager runs the
    app's startup (which creates the database tables)."""
    from main import app

    clients = []

    def _make() -> TestClient:
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


def register(client: TestClient, username: str, password: str = "correct-horse-1") -> dict:
    response = client.post("/api/auth/register", json={"username": username, "password": password})
    assert response.status_code == 201, response.text
    return response.json()
