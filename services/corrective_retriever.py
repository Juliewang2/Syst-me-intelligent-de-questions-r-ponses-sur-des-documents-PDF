"""
services/corrective_retriever.py
----------------------------------
Corrective retrieval, after CRAG (Yan et al., 2024): judge how good a
search was before answering, and search again when it was poor.

CRAG trains an evaluator and falls back to web search. Here the
evaluator is the reranker that already runs (its best 0-10 score), and
the fallback is a second search in the same corpus with rewritten
queries (services/query_rewriter.py): diplomatic questions mostly fail
on wording (abbreviations, jargon) or language (French question,
English text), not because the answer is missing from the corpus.

Best rerank score of the first search:
  CORRECT    (>= upper)        keep the results (filtered, top-k)
  AMBIGUOUS  (lower .. upper)  keep the passages >= lower, add the
                               candidates of every rewrite, fuse (RRF),
                               rerank once against the original question
  INCORRECT  (< lower, or no   same, without the first passages; if
              candidates)      nothing reaches `lower`, return [] so the
                               answer says the documents lack it
No scores at all (rerank off or failed) counts as CORRECT: there is
nothing to judge with, so the plain pipeline's result is kept.
No rewrites -> first-round result. A failed second rerank -> first-round
result. At most `corrective_max_retries` extra rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Collection, List, Optional

from langchain_core.documents import Document as LCDocument

from config import get_settings
from services import retriever
from services.llm_factory import llm_call_log
from services.query_rewriter import corpus_languages, rewrite_query

settings = get_settings()

# Passages shown to the rewriter as "what the failed search found".
_FAILED_CONTEXT = 4

STATUS_MESSAGES = {
    "AMBIGUOUS": "Results look only partly relevant - searching again with reformulated queries...",
    "INCORRECT": "No clearly relevant passage found - searching again with reformulated queries...",
}


class RetrievalAction(str, Enum):
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"


@dataclass
class CorrectiveResult:
    documents: List[LCDocument]
    action: RetrievalAction  # judgement of the first search
    attempts: int  # searches run (1 = no retry)
    rewritten_queries: List[str] = field(default_factory=list)
    max_scores: List[Optional[int]] = field(default_factory=list)  # best score per search
    llm_calls: int = 0
    initial_documents: List[LCDocument] = field(default_factory=list)  # first-round top-k

    def trace(self) -> dict:
        """JSON-friendly summary (no passage text)."""
        return {
            "action": self.action.value,
            "attempts": self.attempts,
            "rewritten_queries": self.rewritten_queries,
            "max_scores": self.max_scores,
            "llm_calls": self.llm_calls,
        }


def lower_threshold() -> int:
    if settings.corrective_lower_threshold is not None:
        return settings.corrective_lower_threshold
    return settings.rerank_min_score


def _score(doc: LCDocument) -> Optional[int]:
    return doc.metadata.get("rerank_score")


def max_score(docs: List[LCDocument]) -> Optional[int]:
    scores = [s for s in map(_score, docs) if s is not None]
    return max(scores) if scores else None


def classify(scored_docs: List[LCDocument]) -> RetrievalAction:
    """Judge a search from its candidates' rerank scores."""
    if not scored_docs:
        return RetrievalAction.INCORRECT
    best = max_score(scored_docs)
    if best is None or best >= settings.corrective_upper_threshold:
        return RetrievalAction.CORRECT
    if best < lower_threshold():
        return RetrievalAction.INCORRECT
    return RetrievalAction.AMBIGUOUS


def _passing(docs: List[LCDocument]) -> List[LCDocument]:
    """Scored passages at or above the lower threshold."""
    return [d for d in docs if _score(d) is not None and _score(d) >= lower_threshold()]


def _search_again(
    owner_id: str,
    question: str,
    rewrites: List[str],
    kept: List[LCDocument],
    document_id: Optional[str],
    filters: Optional[dict],
    hybrid: Optional[bool],
    n_candidates: int,
) -> Optional[List[LCDocument]]:
    """Kept passages + every rewrite's candidates, fused and reranked once
    against the original question. None if that rerank failed."""
    ranked_lists = [kept] + [
        retriever.retrieve_scored(
            owner_id, query, document_id=document_id, filters=filters, hybrid=hybrid,
            use_rerank=False, n_candidates=n_candidates,
        )
        for query in rewrites
    ]
    candidates = retriever.reciprocal_rank_fusion(ranked_lists)[:n_candidates]
    if not candidates:
        return []
    scored = retriever.rerank(question, candidates)
    if scored is None:
        return None
    return [retriever._with_scores(doc, rerank_score=score) for doc, score in scored]  # noqa: SLF001


def corrective_retrieve(
    owner_id: str,
    question: str,
    document_id: Optional[str] = None,
    filters: Optional[dict] = None,
    k: Optional[int] = None,
    rewrite_modes: Optional[Collection[str]] = None,
    on_status: Optional[Callable[[dict], None]] = None,
    hybrid: Optional[bool] = None,
    use_rerank: Optional[bool] = None,
) -> CorrectiveResult:
    """Retrieve for `question`, searching again with rewritten queries
    when the first search scores low. `on_status` is called with
    {"action", "message"} just before a retry starts."""
    k = k or settings.retriever_k
    n_candidates = max(settings.retrieval_candidates, k)
    with llm_call_log() as calls:
        scored = retriever.retrieve_scored(
            owner_id, question, document_id=document_id, filters=filters, hybrid=hybrid,
            use_rerank=use_rerank, n_candidates=n_candidates,
        )
        first = retriever.above_threshold(scored, lower_threshold())[:k]
        action = classify(scored)
        result = CorrectiveResult(
            documents=first, action=action, attempts=1, max_scores=[max_score(scored)], initial_documents=first
        )

        current_action, current = action, scored
        for _ in range(settings.corrective_max_retries):
            if current_action == RetrievalAction.CORRECT:
                break
            if on_status is not None:
                on_status({"action": current_action.value, "message": STATUS_MESSAGES[current_action.value]})
            rewrites = rewrite_query(
                question, corpus_languages(owner_id), failed_context=current[:_FAILED_CONTEXT], modes=rewrite_modes
            )
            if not rewrites:
                break
            kept = _passing(current) if current_action == RetrievalAction.AMBIGUOUS else []
            rescored = _search_again(owner_id, question, rewrites, kept, document_id, filters, hybrid, n_candidates)
            result.attempts += 1
            result.rewritten_queries += rewrites
            if rescored is None:  # second rerank failed: keep what we had
                break
            result.max_scores.append(max_score(rescored))
            result.documents = _passing(rescored)[:k]
            current_action, current = classify(rescored), rescored

    result.llm_calls = sum(c["event"] == "model" for c in calls)
    return result
