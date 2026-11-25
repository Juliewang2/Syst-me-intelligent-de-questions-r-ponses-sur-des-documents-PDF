"""
tests/test_corrective.py
--------------------------
The corrective retrieval loop with every model call mocked: the first
search (retrieve_scored), the rewriter and the second rerank are
replaced by fakes, so each branch (CORRECT / AMBIGUOUS / INCORRECT,
failures, retry limit) is driven by chosen scores.
"""

from typing import Dict, List, Optional

import pytest
from langchain_core.documents import Document as LCDocument

from services import corrective_retriever as cr
from services import retriever
from services.corrective_retriever import RetrievalAction, classify, corrective_retrieve


def _doc(name: str, score: Optional[int] = None) -> LCDocument:
    meta = {"document_id": name, "chunk_index": 0, "page": 1, "page_end": 1, "document_name": f"{name}.pdf"}
    if score is not None:
        meta["rerank_score"] = score
    return LCDocument(page_content=f"text of {name}", metadata=meta)


class Fakes:
    """Scripted search results, rewrites and rerank scores."""

    def __init__(self, monkeypatch, first: List[LCDocument], by_query: Dict[str, List[str]] = None,
                 rewrites=None, scores: Dict[str, int] = None, rerank_fails: bool = False):
        self.first, self.by_query = first, by_query or {}
        self.rewrites = rewrites if rewrites is not None else []
        self.scores, self.rerank_fails = scores or {}, rerank_fails
        self.rewrite_calls, self.rerank_calls, self.searches = 0, 0, []
        monkeypatch.setattr(retriever, "retrieve_scored", self.retrieve_scored)
        monkeypatch.setattr(retriever, "rerank", self.rerank)
        monkeypatch.setattr(cr, "rewrite_query", self.rewrite_query)
        monkeypatch.setattr(cr, "corpus_languages", lambda owner: ["en", "fr"])

    def retrieve_scored(self, owner_id, query, use_rerank=None, **kwargs):
        self.searches.append((query, use_rerank))
        if query == "question":
            return list(self.first)
        return [_doc(name) for name in self.by_query.get(query, [])]

    def rewrite_query(self, question, languages, failed_context=None, modes=None):
        self.rewrite_calls += 1
        self.modes = modes
        if isinstance(self.rewrites, Exception):
            return []  # rewrite_query never raises: it returns []
        return list(self.rewrites)

    def rerank(self, question, docs):
        self.rerank_calls += 1
        assert question == "question"  # always against the original question
        if self.rerank_fails:
            return None
        scored = [(d, self.scores.get(d.metadata["document_id"], 0)) for d in docs]
        return sorted(scored, key=lambda p: p[1], reverse=True)


@pytest.fixture(autouse=True)
def thresholds(monkeypatch):
    monkeypatch.setattr(cr.settings, "corrective_upper_threshold", 7)
    monkeypatch.setattr(cr.settings, "corrective_lower_threshold", None)
    monkeypatch.setattr(cr.settings, "rerank_min_score", 4)
    monkeypatch.setattr(cr.settings, "corrective_max_retries", 1)


def _names(docs):
    return [d.metadata["document_id"] for d in docs]


def test_classify():
    assert classify([]) == RetrievalAction.INCORRECT
    assert classify([_doc("a"), _doc("b")]) == RetrievalAction.CORRECT  # no scores
    assert classify([_doc("a", 8), _doc("b", 1)]) == RetrievalAction.CORRECT
    assert classify([_doc("a", 5)]) == RetrievalAction.AMBIGUOUS
    assert classify([_doc("a", 3)]) == RetrievalAction.INCORRECT


def test_correct_keeps_first_results_without_rewriting(monkeypatch):
    fakes = Fakes(monkeypatch, first=[_doc("a", 9), _doc("b", 5), _doc("c", 2)])
    result = corrective_retrieve("owner", "question")
    assert result.action == RetrievalAction.CORRECT and result.attempts == 1
    assert _names(result.documents) == ["a", "b"]  # c below the threshold
    assert fakes.rewrite_calls == 0


def test_all_scores_none_counts_as_correct(monkeypatch):
    fakes = Fakes(monkeypatch, first=[_doc("a"), _doc("b")])
    result = corrective_retrieve("owner", "question")
    assert result.action == RetrievalAction.CORRECT and _names(result.documents) == ["a", "b"]
    assert fakes.rewrite_calls == 0


def test_ambiguous_keeps_passing_passages_and_adds_rewrite_results(monkeypatch):
    statuses = []
    fakes = Fakes(
        monkeypatch,
        first=[_doc("a", 5), _doc("b", 2)],
        rewrites=["expanded question"],
        by_query={"expanded question": ["c", "a"]},
        scores={"a": 6, "c": 9, "b": 8},
    )
    result = corrective_retrieve("owner", "question", on_status=statuses.append)

    assert result.action == RetrievalAction.AMBIGUOUS and result.attempts == 2
    assert _names(result.documents) == ["c", "a"]  # b (below lower) was not kept
    assert result.rewritten_queries == ["expanded question"] and result.max_scores == [5, 9]
    assert ("expanded question", False) in fakes.searches  # rewrites searched without rerank
    assert fakes.rerank_calls == 1
    assert statuses == [{"action": "AMBIGUOUS", "message": cr.STATUS_MESSAGES["AMBIGUOUS"]}]
    assert _names(result.initial_documents) == ["a"]


def test_incorrect_drops_first_passages(monkeypatch):
    fakes = Fakes(
        monkeypatch,
        first=[_doc("a", 3)],
        rewrites=["translated"],
        by_query={"translated": ["d"]},
        scores={"a": 9, "d": 8},
    )
    result = corrective_retrieve("owner", "question")
    assert result.action == RetrievalAction.INCORRECT
    assert _names(result.documents) == ["d"]  # a was not carried over


def test_incorrect_returns_nothing_when_rewrites_find_nothing_relevant(monkeypatch):
    Fakes(monkeypatch, first=[_doc("a", 1)], rewrites=["translated"], by_query={"translated": ["d"]},
          scores={"d": 2})
    assert corrective_retrieve("owner", "question").documents == []


def test_no_candidates_at_all_is_incorrect(monkeypatch):
    Fakes(monkeypatch, first=[], rewrites=["x"], by_query={"x": []})
    result = corrective_retrieve("owner", "question")
    assert result.action == RetrievalAction.INCORRECT and result.documents == []


def test_rewrite_failure_means_no_retry(monkeypatch):
    fakes = Fakes(monkeypatch, first=[_doc("a", 5)], rewrites=RuntimeError("model down"))
    result = corrective_retrieve("owner", "question")
    assert result.attempts == 1 and _names(result.documents) == ["a"]
    assert fakes.rerank_calls == 0


def test_failed_second_rerank_keeps_first_round(monkeypatch):
    Fakes(monkeypatch, first=[_doc("a", 5)], rewrites=["x"], by_query={"x": ["c"]}, rerank_fails=True)
    result = corrective_retrieve("owner", "question")
    assert _names(result.documents) == ["a"] and result.attempts == 2


@pytest.mark.parametrize("max_retries", [1, 2])
def test_retry_limit(monkeypatch, max_retries):
    monkeypatch.setattr(cr.settings, "corrective_max_retries", max_retries)
    fakes = Fakes(monkeypatch, first=[_doc("a", 5)], rewrites=["x"], by_query={"x": ["a"]}, scores={"a": 5})
    result = corrective_retrieve("owner", "question")
    assert result.attempts == 1 + max_retries and fakes.rewrite_calls == max_retries


def test_merge_has_no_duplicates(monkeypatch):
    Fakes(monkeypatch, first=[_doc("a", 5), _doc("b", 6)], rewrites=["x", "y"],
          by_query={"x": ["a", "c"], "y": ["c", "b", "a"]}, scores={"a": 8, "b": 8, "c": 8})
    result = corrective_retrieve("owner", "question")
    keys = [(d.metadata["document_id"], d.metadata["chunk_index"]) for d in result.documents]
    assert len(keys) == len(set(keys)) == 3


def test_rewrite_modes_are_passed_on(monkeypatch):
    fakes = Fakes(monkeypatch, first=[_doc("a", 5)], rewrites=[])
    corrective_retrieve("owner", "question", rewrite_modes={"expand"})
    assert fakes.modes == {"expand"}
