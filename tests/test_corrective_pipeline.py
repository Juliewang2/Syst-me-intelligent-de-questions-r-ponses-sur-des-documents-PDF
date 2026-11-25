"""
tests/test_corrective_pipeline.py
-----------------------------------
Corrective retrieval wired into the RAG service: switched off it changes
nothing; switched on, its trace is returned and its status events are
streamed before the sources (statuses cross from the retrieval thread
to the event loop).
"""

import asyncio

import pytest
from langchain_core.documents import Document as LCDocument

from services import llm_factory, rag_service
from services.corrective_retriever import CorrectiveResult, RetrievalAction
from services.retriever import retrieve
from services.vector_store import add_document_chunks
from tests.test_leak import RecordingChat


def _chunk(text: str, index: int) -> LCDocument:
    return LCDocument(page_content=text, metadata={
        "document_id": "d", "document_name": "d.pdf", "page": 1, "page_end": 1, "chunk_index": index,
    })


@pytest.fixture
def fake_models(monkeypatch):
    model = RecordingChat(provider="openai")
    monkeypatch.setitem(llm_factory.CHAT_BUILDERS, "openai", lambda *a: model)
    llm_factory.reset_cache()
    yield model
    llm_factory.reset_cache()


def test_switched_off_retrieval_is_unchanged(fake_embeddings, monkeypatch):
    monkeypatch.setattr(rag_service.settings, "corrective_retrieval_enabled", False)
    add_document_chunks("owner", [_chunk(f"apple pie recipe {i}", i) for i in range(6)] + [_chunk("zebra", 6)])

    result = rag_service.retrieve_context("apple pie recipe", [], "owner")
    expected = retrieve("owner", "apple pie recipe")
    assert [d.page_content for d in result.documents] == [d.page_content for d in expected]
    assert [d.metadata for d in result.documents] == [d.metadata for d in expected]
    assert result.trace is None


@pytest.fixture
def scripted_corrective(monkeypatch):
    """corrective_retrieve replaced by a fake that reports two statuses
    from the worker thread, then returns one document."""
    doc = _chunk("relevant passage", 0)
    # Labeled, so the generation model is the (fake) external one whatever .env says.
    doc.metadata["classification"] = "PUBLIC"

    def fake(owner_id, question, on_status=None, **kwargs):
        if on_status is not None:
            on_status({"action": "AMBIGUOUS", "message": "searching again"})
            on_status({"action": "AMBIGUOUS", "message": "still searching"})
        return CorrectiveResult(documents=[doc], action=RetrievalAction.AMBIGUOUS, attempts=2,
                                rewritten_queries=["rewritten"], max_scores=[5, 9], llm_calls=2)

    monkeypatch.setattr(rag_service.settings, "corrective_retrieval_enabled", True)
    monkeypatch.setattr(rag_service, "corrective_retrieve", fake)


def test_trace_is_returned_when_enabled(fake_models, scripted_corrective):
    answer, sources, trace = rag_service.generate_answer_traced("question", [], "owner")
    assert trace == {"action": "AMBIGUOUS", "attempts": 2, "rewritten_queries": ["rewritten"],
                     "max_scores": [5, 9], "llm_calls": 2}
    assert [s["text_snippet"] for s in sources] == ["relevant passage"]


def test_status_events_stream_before_sources(fake_models, scripted_corrective):
    async def collect():
        return [e async for e in rag_service.generate_answer_stream("question", [], "owner")]

    events = asyncio.run(collect())
    types = [e["type"] for e in events]
    assert types[:3] == ["status", "status", "sources"]
    assert [e["message"] for e in events[:2]] == ["searching again", "still searching"]
    assert events[-1]["type"] == "done" and events[-1]["retrieval_trace"]["attempts"] == 2


def test_chat_endpoint_returns_trace(fake_models, scripted_corrective, make_client):
    from tests.conftest import register

    client = make_client()
    register(client, "trace-user")
    response = client.post("/api/chat", json={"message": "question"})
    assert response.status_code == 200, response.text
    assert response.json()["retrieval_trace"]["action"] == "AMBIGUOUS"

    stream = client.post("/api/chat/stream", json={"message": "question"}).text
    assert stream.index('"type": "status"') < stream.index('"type": "sources"')
