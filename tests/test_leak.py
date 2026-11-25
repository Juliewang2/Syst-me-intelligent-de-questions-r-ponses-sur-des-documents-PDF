"""
tests/test_leak.py
--------------------
Leak probe for the hybrid deployment mode: the external providers
(OpenAI, Mistral) are replaced by a fake chat model that records every
input it receives, Ollama by another one. The full pipeline (question
condensation, retrieval with LLM rerank, generation - non-streaming and
streaming) then runs on an index holding a restricted and a public
chunk. Restricted text must never reach the external fake, and public
requests must really go to it (otherwise the test would pass vacuously).
"""

import asyncio
from typing import Any, List, Optional

import pytest
from langchain_core.documents import Document as LCDocument
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from services import llm_factory, rag_service, retriever
from services.sensitivity import HISTORY_KEY
from services.vector_store import add_document_chunks

SECRET = "Valmont instructs its delegation to vote against draft XJ9 quietly"
PUBLIC_TEXT = "Resolution 77/276 requests an advisory opinion of the Court on climate obligations"
SECRET_ANSWER = "Valmont will vote against XJ9"


class RecordingChat(BaseChatModel):
    """Fake chat model that logs every prompt text it is given."""

    provider: str
    log: List[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return f"recording-{self.provider}"

    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs: Any) -> dict:
        return {"ls_provider": self.provider, "ls_model_name": "fake"}

    def _record(self, messages: List[BaseMessage]) -> None:
        self.log.append("\n".join(str(m.content) for m in messages))

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self._record(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="fake answer"))])

    def with_structured_output(self, schema, **kwargs: Any):
        def run(value):
            self._record(value.to_messages() if hasattr(value, "to_messages") else list(value))
            # Every passage relevant, so all of them reach generation.
            return schema.model_validate({"scores": [{"index": i, "score": 10} for i in range(1, 30)]})

        return RunnableLambda(run)


@pytest.fixture
def probes(fake_embeddings, monkeypatch):
    external = RecordingChat(provider="openai")
    local = RecordingChat(provider="ollama")
    monkeypatch.setitem(llm_factory.CHAT_BUILDERS, "openai", lambda *a: external)
    monkeypatch.setitem(llm_factory.CHAT_BUILDERS, "mistral", lambda *a: external)
    monkeypatch.setitem(llm_factory.CHAT_BUILDERS, "ollama", lambda *a: local)
    monkeypatch.setattr(llm_factory.settings, "deployment_mode", "hybride")
    monkeypatch.setattr(retriever.settings, "rerank_enabled", True)
    llm_factory.reset_cache()

    meta = {"document_name": "x.pdf", "page": 1, "page_end": 1}
    add_document_chunks(
        "owner",
        [
            LCDocument(page_content=SECRET, metadata={**meta, "document_id": "dr", "chunk_index": 0,
                                                      "classification": "DR_SIMULE"}),
            LCDocument(page_content=PUBLIC_TEXT, metadata={**meta, "document_id": "pub", "chunk_index": 0,
                                                           "classification": "PUBLIC"}),
        ],
    )
    yield external, local
    llm_factory.reset_cache()


def _leaked(model: RecordingChat, *texts: str) -> bool:
    return any(text in entry for entry in model.log for text in texts)


def _stream(question: str, history: List[BaseMessage]) -> List[dict]:
    async def collect():
        return [e async for e in rag_service.generate_answer_stream(question, history, "owner")]

    return asyncio.run(collect())


def test_restricted_context_never_reaches_external_model(probes):
    external, local = probes
    answer, sources = rag_service.generate_answer("How will Valmont vote on draft XJ9?", [], "owner")

    assert {s["classification"] for s in sources} >= {"DR_SIMULE"}
    assert not _leaked(external, SECRET)
    assert _leaked(local, SECRET)  # rerank and generation ran, locally


def test_restricted_context_never_leaks_when_streaming(probes):
    external, local = probes
    events = _stream("How will Valmont vote on draft XJ9?", [])

    assert events[-1]["type"] == "done"
    assert not _leaked(external, SECRET)
    assert _leaked(local, SECRET)


def test_restricted_history_stays_local(probes):
    external, _ = probes
    history = [
        HumanMessage(content="How will Valmont vote?"),
        AIMessage(content=SECRET_ANSWER, response_metadata={HISTORY_KEY: "DR_SIMULE"}),
    ]
    rag_service.generate_answer("And what does resolution 77/276 request?", history, "owner")
    _stream("And what does resolution 77/276 request?", history)

    assert not _leaked(external, SECRET, SECRET_ANSWER)


def test_public_request_goes_to_external_model(probes):
    external, local = probes
    answer, sources = rag_service.generate_answer("What does resolution 77/276 request?", [], "owner")

    assert {s["classification"] for s in sources} == {"PUBLIC"}
    assert _leaked(external, PUBLIC_TEXT)  # rerank + generation went external
    assert not _leaked(external, SECRET)
    assert local.log == []


def test_public_history_is_condensed_externally(probes):
    external, _ = probes
    history = [
        HumanMessage(content="What is 77/276?"),
        AIMessage(content="A resolution.", response_metadata={HISTORY_KEY: "PUBLIC"}),
    ]
    rag_service.generate_answer("What does it request?", history, "owner")
    assert _leaked(external, "Follow-up question: What does it request?")
