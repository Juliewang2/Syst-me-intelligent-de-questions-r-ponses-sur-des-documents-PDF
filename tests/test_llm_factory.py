"""
tests/test_llm_factory.py
---------------------------
Provider choice per deployment mode / role / sensitivity, the call log,
and structured() - method per provider, schema in the Ollama prompt,
one retry, StructuredOutputError - all offline (models are built but
never called; with_structured_output is replaced by fakes).
"""

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from services import llm_factory
from services.llm_factory import (
    StructuredOutputError,
    embedding_choice,
    get_chat_model,
    llm_call_log,
    resolve_chat_model,
    structured,
)


class Answer(BaseModel):
    value: int


@pytest.fixture(autouse=True)
def _fresh_factory(monkeypatch):
    llm_factory.reset_cache()
    yield
    llm_factory.reset_cache()


def _set(monkeypatch, **values):
    for key, value in values.items():
        monkeypatch.setattr(llm_factory.settings, key, value)


# --- provider choice -------------------------------------------------------

@pytest.mark.parametrize(
    "mode, provider",
    [("openai", "openai"), ("mistral", "mistral"), ("souverain", "ollama"), ("hybride", "openai")],
)
def test_mode_sets_default_chat_provider(monkeypatch, mode, provider):
    _set(monkeypatch, deployment_mode=mode)
    assert resolve_chat_model("generate").provider == provider


def test_role_override_wins_over_mode(monkeypatch):
    _set(monkeypatch, deployment_mode="hybride", rerank_provider="mistral", rerank_model="mistral-large")
    choice = resolve_chat_model("rerank")
    assert (choice.provider, choice.model) == ("mistral", "mistral-large")
    assert resolve_chat_model("generate").provider == "openai"


def test_restricted_context_is_rerouted_to_local(monkeypatch):
    _set(monkeypatch, deployment_mode="hybride", generate_model="gpt-4o")
    choice = resolve_chat_model("generate", "DR_SIMULE")
    assert choice.rerouted and choice.provider == "ollama"
    # The override named an external model; the local default replaces it.
    assert choice.model == llm_factory.settings.local_chat_model


def test_allowed_provider_is_not_rerouted(monkeypatch):
    _set(monkeypatch, deployment_mode="mistral", dr_allowed_providers=["ollama", "mistral"])
    choice = resolve_chat_model("generate", "DR_SIMULE")
    assert not choice.rerouted and choice.provider == "mistral"


def test_unknown_role_or_provider_is_an_error(monkeypatch):
    with pytest.raises(ValueError):
        resolve_chat_model("summarize")
    _set(monkeypatch, generate_provider="anthropic")
    with pytest.raises(ValueError):
        resolve_chat_model("generate")


def test_call_log_records_provider_and_sensitivity(monkeypatch):
    _set(monkeypatch, deployment_mode="hybride")
    with llm_call_log() as calls:
        get_chat_model("generate", "PUBLIC")
        get_chat_model("rerank", "DR_SIMULE")
    assert [(c["role"], c["provider"], c["sensitivity"], c["rerouted"]) for c in calls] == [
        ("generate", "openai", "PUBLIC", False),
        ("rerank", "ollama", "DR_SIMULE", True),
    ]
    get_chat_model("generate")  # outside the block: not recorded, no error
    assert len(calls) == 2


@pytest.mark.parametrize(
    "mode, provider", [("openai", "openai"), ("mistral", "mistral"), ("hybride", "ollama"), ("souverain", "ollama")]
)
def test_embeddings_are_local_in_hybrid_and_sovereign_modes(monkeypatch, mode, provider):
    _set(monkeypatch, deployment_mode=mode)
    assert embedding_choice()[0] == provider


def test_local_embeddings_are_unit_length():
    class Raw:
        def embed_documents(self, texts):
            return [[3.0, 4.0] for _ in texts]

        def embed_query(self, text):
            return [0.0, 2.0]

    wrapped = llm_factory.NormalizedEmbeddings(Raw())
    assert wrapped.embed_documents(["x"]) == [[0.6, 0.8]]
    assert wrapped.embed_query("x") == [0.0, 1.0]


# --- structured() ----------------------------------------------------------

def _model(provider: str):
    return llm_factory.CHAT_BUILDERS[provider]("some-model", 0.3, False, 10.0, 0)


def _fake_structured(monkeypatch, model_cls, outputs, seen):
    """Replace model_cls.with_structured_output: records the method and
    the messages it receives, then returns `outputs` one by one (an
    exception instance is raised instead)."""
    queue = list(outputs)

    def with_structured_output(self, schema, *, method="function_calling", **kwargs):
        seen.append({"method": method, "strict": kwargs.get("strict"), "temperature": self.temperature})

        def run(value):
            seen[-1]["messages"] = value.to_messages() if hasattr(value, "to_messages") else value
            result = queue.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        return RunnableLambda(run)

    monkeypatch.setattr(model_cls, "with_structured_output", with_structured_output)


@pytest.mark.parametrize(
    "provider, method, strict",
    [("openai", "json_schema", True), ("mistral", "function_calling", None), ("ollama", "json_schema", None)],
)
def test_structured_picks_method_per_provider(monkeypatch, provider, method, strict):
    llm = _model(provider)
    seen = []
    _fake_structured(monkeypatch, type(llm), [Answer(value=1)], seen)
    assert structured(llm, Answer).invoke("q") == Answer(value=1)
    assert (seen[0]["method"], seen[0]["strict"]) == (method, strict)


def test_ollama_gets_temperature_zero_and_schema_in_system_prompt(monkeypatch):
    llm = _model("ollama")
    seen = []
    _fake_structured(monkeypatch, type(llm), [Answer(value=2)], seen)
    prompt = ChatPromptTemplate.from_messages([("system", "Judge passages."), ("human", "{q}")])
    (prompt | structured(llm, Answer)).invoke({"q": "hello"})

    assert seen[0]["temperature"] == 0
    system, human = seen[0]["messages"]
    assert isinstance(system, SystemMessage) and system.content.startswith("Judge passages.")
    assert '"value"' in system.content and "JSON schema" in system.content
    assert isinstance(human, HumanMessage) and human.content == "hello"
    assert llm.temperature == 0.3  # the shared model itself is unchanged


def test_structured_retries_once_after_parse_failure(monkeypatch):
    llm = _model("openai")
    seen = []
    _fake_structured(monkeypatch, type(llm), [OutputParserException("bad json"), Answer(value=3)], seen)
    assert structured(llm, Answer).invoke("q") == Answer(value=3)


def test_structured_raises_after_second_failure(monkeypatch):
    llm = _model("mistral")
    seen = []
    _fake_structured(monkeypatch, type(llm), [OutputParserException("bad"), None], seen)
    with llm_call_log() as calls:
        with pytest.raises(StructuredOutputError):
            structured(llm, Answer).invoke("q")
    assert [c["event"] for c in calls] == ["structured_call", "structured_failure"]
    assert calls[-1] == {"event": "structured_failure", "provider": "mistral", "schema": "Answer"}


def test_structured_does_not_retry_network_errors(monkeypatch):
    llm = _model("openai")
    seen = []
    _fake_structured(monkeypatch, type(llm), [TimeoutError("slow"), Answer(value=4)], seen)
    with pytest.raises(TimeoutError):
        structured(llm, Answer).invoke("q")
