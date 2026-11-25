"""
tests/test_sensitivity.py
---------------------------
The sensitivity rules (services/sensitivity.py) and the history marking
done by conversation_memory.load_chat_history.
"""

import json

import pytest
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessage, HumanMessage

from services import sensitivity
from services.sensitivity import (
    context_sensitivity,
    history_sensitivity,
    sources_sensitivity,
    stricter,
)


def _doc(classification=None) -> LCDocument:
    meta = {} if classification is None else {"classification": classification}
    return LCDocument(page_content="text", metadata=meta)


@pytest.fixture
def metadata_mode(monkeypatch):
    """Hybrid mode with corpus metadata on (the DiploDoc setup)."""
    monkeypatch.setattr(sensitivity.settings, "deployment_mode", "hybride")
    monkeypatch.setattr(sensitivity.settings, "corpus_metadata_enabled", True)


def test_any_restricted_chunk_makes_the_context_restricted(metadata_mode):
    assert context_sensitivity([_doc("PUBLIC"), _doc("PUBLIC")]) == "PUBLIC"
    assert context_sensitivity([_doc("PUBLIC"), _doc("DR_SIMULE")]) == "DR_SIMULE"
    assert context_sensitivity([]) == "PUBLIC"


def test_unlabeled_chunk_is_restricted_by_default(metadata_mode):
    assert context_sensitivity([_doc(None)]) == "DR_SIMULE"
    assert sources_sensitivity([{"classification": None}]) == "DR_SIMULE"


def test_unlabeled_chunk_is_public_in_legacy_openai_setup(monkeypatch):
    # Default configuration: nothing is classified, behaviour unchanged.
    monkeypatch.setattr(sensitivity.settings, "deployment_mode", "openai")
    monkeypatch.setattr(sensitivity.settings, "corpus_metadata_enabled", False)
    assert context_sensitivity([_doc(None)]) == "PUBLIC"
    assert context_sensitivity([_doc("DR_SIMULE")]) == "DR_SIMULE"


def test_stricter_and_history():
    assert stricter("PUBLIC", None) == "PUBLIC"
    assert stricter("PUBLIC", "DR_SIMULE") == "DR_SIMULE"
    history = [HumanMessage(content="q"), AIMessage(content="a", response_metadata={"sensitivity": "PUBLIC"})]
    assert history_sensitivity(history) == "PUBLIC"
    history.append(AIMessage(content="b", response_metadata={"sensitivity": "DR_SIMULE"}))
    assert history_sensitivity(history) == "DR_SIMULE"


def test_load_chat_history_marks_answers_from_restricted_sources(metadata_mode):
    from database import db_session, init_db
    from models import ChatMessage, Conversation, User
    from services.conversation_memory import load_chat_history

    init_db()
    with db_session() as db:
        user = User(username="history-owner", password_hash="x")
        db.add(user)
        db.commit()
        conversation = Conversation(owner_id=user.id, title="t")
        db.add(conversation)
        db.commit()
        db.add_all(
            [
                ChatMessage(conversation_id=conversation.id, role="user", content="q1"),
                ChatMessage(conversation_id=conversation.id, role="assistant", content="a1",
                            sources=json.dumps([{"classification": "PUBLIC"}])),
                ChatMessage(conversation_id=conversation.id, role="assistant", content="a2",
                            sources=json.dumps([{"classification": "DR_SIMULE"}])),
            ]
        )
        db.commit()
        history = load_chat_history(db, conversation.id)

    marks = [m.response_metadata.get("sensitivity") for m in history if isinstance(m, AIMessage)]
    assert sorted(marks) == ["DR_SIMULE", "PUBLIC"]
    assert history_sensitivity(history) == "DR_SIMULE"
