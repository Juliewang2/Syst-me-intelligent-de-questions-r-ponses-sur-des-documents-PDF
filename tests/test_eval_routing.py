"""
tests/test_eval_routing.py
----------------------------
Routing metrics of evaluation/run_eval.py: per-question fields from the
llm_factory call log, the summary rates, --deployment /
--allow-external-dr, and a question run end to end in hybrid mode.
"""

import pytest
from langchain_core.documents import Document as LCDocument

from evaluation import run_eval
from evaluation.run_eval import apply_deployment, routing_rates, routing_summary
from services import llm_factory, retriever
from services.vector_store import add_document_chunks
from tests.test_leak import RecordingChat


def _model(role, provider="openai", rerouted=False):
    return {"event": "model", "role": role, "provider": provider, "model": "m",
            "sensitivity": "DR_SIMULE" if rerouted else "PUBLIC", "rerouted": rerouted}


def test_routing_summary_from_call_log():
    events = [
        _model("rerank", "ollama", rerouted=True),
        {"event": "structured_call", "provider": "ollama", "schema": "_RerankResult"},
        {"event": "structured_failure", "provider": "ollama", "schema": "_RerankResult"},
        {"event": "rerank_fallback", "error": "StructuredOutputError"},
        _model("generate", "ollama", rerouted=True),
    ]
    row = routing_summary(events)
    assert [r["role"] for r in row["llm_requests"]] == ["rerank", "generate"]
    assert "event" not in row["llm_requests"][0]
    assert (row["rerouted"], row["structured_calls"], row["structured_failures"]) == (True, 1, 1)
    assert (row["reranked"], row["rerank_fallback"]) == (True, True)


def test_routing_rates():
    rows = [
        routing_summary([_model("rerank", "ollama", True), {"event": "structured_call"},
                         {"event": "structured_failure"}, {"event": "rerank_fallback"}]),
        routing_summary([_model("rerank"), {"event": "structured_call"}]),
        routing_summary([_model("generate")]),  # no rerank in this config
    ]
    assert routing_rates(rows) == {
        "rerouted_rate": 0.333,
        "structured_failure_rate": 0.5,
        "rerank_fallback_rate": 0.5,
    }
    assert routing_rates([]) == {"rerouted_rate": None, "structured_failure_rate": None,
                                 "rerank_fallback_rate": None}


def test_apply_deployment_and_external_dr_flag(monkeypatch, capsys):
    settings = run_eval.settings
    monkeypatch.setattr(settings, "deployment_mode", settings.deployment_mode)
    monkeypatch.setattr(settings, "dr_allowed_providers", list(settings.dr_allowed_providers))

    apply_deployment("hybride", allow_external_dr=False)
    assert run_eval.deployment_report(False)["embedding"].startswith("ollama/")
    assert llm_factory.resolve_chat_model("generate", "DR_SIMULE").rerouted

    apply_deployment("hybride", allow_external_dr=True)
    assert "EXTERNAL PROVIDERS" in capsys.readouterr().out
    assert not llm_factory.resolve_chat_model("generate", "DR_SIMULE").rerouted
    assert run_eval.deployment_report(True)["external_dr_allowed"] is True


def test_restricted_question_is_rerouted_in_hybrid_mode(fake_embeddings, monkeypatch):
    external, local = RecordingChat(provider="openai"), RecordingChat(provider="ollama")
    monkeypatch.setitem(llm_factory.CHAT_BUILDERS, "openai", lambda *a: external)
    monkeypatch.setitem(llm_factory.CHAT_BUILDERS, "ollama", lambda *a: local)
    monkeypatch.setattr(llm_factory.settings, "deployment_mode", "hybride")
    monkeypatch.setattr(retriever.settings, "rerank_enabled", True)
    llm_factory.reset_cache()
    add_document_chunks("owner", [LCDocument(
        page_content="Valmont instruction on draft XJ9",
        metadata={"document_id": "dr", "document_name": "td.pdf", "page": 1, "page_end": 1,
                  "chunk_index": 0, "classification": "DR_SIMULE"},
    )])

    item = {"question": "Valmont instruction on XJ9?", "reference_answer": "x"}
    docs, answer, _, _, routing = run_eval._run_system(
        "owner", item, {"hybrid": True, "use_rerank": True}, k=4, generate=True
    )
    assert docs and answer == "fake answer"
    assert routing["rerouted"] and {r["provider"] for r in routing["llm_requests"]} == {"ollama"}
    assert external.log == []
    llm_factory.reset_cache()
