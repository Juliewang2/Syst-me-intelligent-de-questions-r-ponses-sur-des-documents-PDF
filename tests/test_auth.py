"""
tests/test_auth.py
--------------------
Authentication and per-user data isolation.
"""

import uuid

from database import db_session
from models import Document
from services.auth_service import hash_password, verify_password
from tests.conftest import register


def _name() -> str:
    return f"user_{uuid.uuid4().hex[:8]}"


def test_password_hashing_roundtrip():
    stored = hash_password("s3cret-pass")
    assert "s3cret-pass" not in stored
    assert verify_password("s3cret-pass", stored)
    assert not verify_password("wrong-pass", stored)
    # Same password, different salt -> different hash.
    assert hash_password("s3cret-pass") != stored


def test_api_requires_login(make_client):
    client = make_client()
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/conversations").status_code == 401
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/api/auth/me").status_code == 401


def test_pages_redirect_to_login(make_client):
    client = make_client()
    response = client.get("/chat", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login?next=/chat"


def test_register_login_logout(make_client):
    client = make_client()
    username = _name()
    register(client, username)
    assert client.get("/api/auth/me").json()["username"] == username

    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401

    bad = client.post("/api/auth/login", json={"username": username, "password": "nope-nope"})
    assert bad.status_code == 401

    good = client.post("/api/auth/login", json={"username": username, "password": "correct-horse-1"})
    assert good.status_code == 200
    assert client.get("/api/auth/me").status_code == 200


def test_bearer_token_works_without_cookie(make_client):
    token = register(make_client(), _name())["access_token"]
    api_client = make_client()  # fresh client: no cookie
    response = api_client.get("/api/documents", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_duplicate_username_rejected(make_client):
    username = _name()
    register(make_client(), username)
    response = make_client().post(
        "/api/auth/register", json={"username": username, "password": "another-pass-1"}
    )
    assert response.status_code == 409


def test_weak_password_rejected(make_client):
    response = make_client().post("/api/auth/register", json={"username": _name(), "password": "short"})
    assert response.status_code == 422


def test_users_cannot_see_each_others_data(make_client):
    alice, bob = make_client(), make_client()
    alice_id = register(alice, _name())["user"]["id"]
    register(bob, _name())

    # A document owned by Alice (inserted directly: uploading needs OpenAI).
    with db_session() as db:
        doc = Document(
            owner_id=alice_id,
            filename="x.pdf",
            original_filename="alice.pdf",
            file_path="/nonexistent/x.pdf",
            status="ready",
        )
        db.add(doc)
        db.flush()
        doc_id = doc.id

    conv_id = alice.post("/api/conversations", json={"title": "Alice's chat"}).json()["id"]

    assert alice.get(f"/api/documents/{doc_id}").status_code == 200
    assert [d["id"] for d in alice.get("/api/documents").json()["documents"]] == [doc_id]

    # Bob sees none of it, and can't touch it by id.
    assert bob.get("/api/documents").json()["total"] == 0
    assert bob.get("/api/conversations").json()["total"] == 0
    assert bob.get(f"/api/documents/{doc_id}").status_code == 404
    assert bob.delete(f"/api/documents/{doc_id}").status_code == 404
    assert bob.get(f"/api/conversations/{conv_id}/messages").status_code == 404
    assert bob.delete(f"/api/conversations/{conv_id}").status_code == 404
    # ...nor chat against Alice's document or conversation.
    assert bob.post("/api/chat", json={"message": "hi", "document_id": doc_id}).status_code == 404
    assert bob.post("/api/chat", json={"message": "hi", "conversation_id": conv_id}).status_code == 404
