"""
tests/test_eval_crag.py
-------------------------
Corrective-retrieval configs in evaluation/run_eval.py: the trace and
first-search hit recorded per question, trigger_rate and rescue_rate.
"""

from langchain_core.documents import Document as LCDocument

from evaluation import run_eval
from evaluation.run_eval import CONFIGS, corrective_rates
from services.corrective_retriever import CorrectiveResult, RetrievalAction


def _doc(doc_id: str, page: int) -> LCDocument:
    return LCDocument(page_content="x", metadata={"doc_id": doc_id, "page": page, "page_end": page})


def _row(action, initial_hit=None, hit=None):
    row = {"retrieval_trace": {"action": action}, "initial_hit": initial_hit}
    if hit is not None:
        row["hit"] = hit
    return row


def test_corrective_rates():
    rows = [
        _row("CORRECT", initial_hit=True, hit=True),
        _row("AMBIGUOUS", initial_hit=False, hit=True),   # rescued
        _row("INCORRECT", initial_hit=False, hit=False),  # not rescued
        _row("AMBIGUOUS", initial_hit=True, hit=True),    # triggered but nothing to rescue
        _row("INCORRECT", initial_hit=None),              # unanswerable
    ]
    assert corrective_rates(rows) == {"trigger_rate": 0.8, "rescue_rate": 0.5}
    assert corrective_rates([{"hit": True}]) == {"trigger_rate": None, "rescue_rate": None}


def test_crag_config_records_trace_and_first_search_hit(monkeypatch):
    seen = {}

    def fake_corrective(owner_id, question, k=None, hybrid=None, use_rerank=None, rewrite_modes=None):
        seen["modes"] = rewrite_modes
        return CorrectiveResult(
            documents=[_doc("a-res-77-276-en", 3)], action=RetrievalAction.INCORRECT, attempts=2,
            rewritten_queries=["q2"], max_scores=[2, 8], llm_calls=3,
            initial_documents=[_doc("a-res-80-138-en", 1)],
        )

    monkeypatch.setattr(run_eval, "corrective_retrieve", fake_corrective)
    item = {"question": "q", "reference_answer": "a",
            "evidence_groups": [{"doc": "a-res-77-276-en", "pages": [3]}]}

    row = run_eval.evaluate_question("owner", item, CONFIGS["hybrid+rerank+crag-noLang"], 4, None, {})
    assert seen["modes"] == {"expand"}
    assert row["retrieval_trace"]["action"] == "INCORRECT" and row["retrieval_trace"]["attempts"] == 2
    assert row["initial_hit"] is False and row["hit"] is True  # rescued


def test_crag_configs_are_not_run_by_default():
    defaults = [name for name, config in CONFIGS.items() if not config.get("crag")]
    assert defaults == ["vector", "hybrid", "hybrid+rerank"]
    assert {"hybrid+rerank+crag", "hybrid+rerank+crag-noLang"} <= set(CONFIGS)
