"""
services/reranker.py
-----------------------
Reranking: a second, more careful relevance pass over the candidates
that fast retrieval returned.

Vector search compares two pre-computed embeddings, which is fast but
coarse - the question and the chunk never "see" each other. A reranker
reads the question together with each candidate and judges relevance
directly, which is much more accurate but too slow to run over a whole
index. So retrieval casts a wide net (~12 candidates) and the reranker
picks the best few.

Here the chat model itself acts as the reranker: it scores every
candidate 0-10 in one call, returning JSON constrained by a schema
(llm_factory.structured(), whose method depends on the provider). The
model is chosen for the candidates' sensitivity, so restricted passages
are scored by a local model. Candidates below `RERANK_MIN_SCORE` are dropped,
which also acts as a relevance threshold: if nothing is relevant, the
answer step is told so instead of being fed noise.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document as LCDocument
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from config import get_settings
from services.llm_factory import get_chat_model, record_event, structured
from services.prompt_templates import RERANK_PROMPT
from services.sensitivity import context_sensitivity

logger = logging.getLogger(__name__)
settings = get_settings()

# Long chunks are shortened in the rerank prompt to bound its cost.
_MAX_PASSAGE_CHARS = 1200


class _PassageScore(BaseModel):
    index: int = Field(description="The passage number, as shown in [Passage N].")
    score: int = Field(description="Relevance from 0 (unrelated) to 10 (directly answers it).")


class _RerankResult(BaseModel):
    scores: List[_PassageScore]


def _get_rerank_chain(docs: List[LCDocument]) -> Runnable:
    llm = get_chat_model("rerank", context_sensitivity(docs), temperature=0.0)
    return RERANK_PROMPT | structured(llm, _RerankResult)


def _format_passages(docs: List[LCDocument]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        text = doc.page_content.strip()
        if len(text) > _MAX_PASSAGE_CHARS:
            text = text[:_MAX_PASSAGE_CHARS] + " ..."
        parts.append(f"[Passage {i}]\n{text}")
    return "\n\n".join(parts)


def rerank(question: str, docs: List[LCDocument]) -> Optional[List[Tuple[LCDocument, int]]]:
    """Score each doc 0-10 for relevance to the question and return
    (doc, score) pairs sorted best first. Returns None if the model call
    fails, so callers can fall back to the original order."""
    if not docs:
        return []
    try:
        result: _RerankResult = _get_rerank_chain(docs).invoke(
            {"question": question, "passages": _format_passages(docs)}
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranking failed, keeping retrieval order: %s", exc)
        record_event("rerank_fallback", error=type(exc).__name__)
        return None

    scores = {s.index: max(0, min(10, s.score)) for s in result.scores}
    scored = [(doc, scores.get(i, 0)) for i, doc in enumerate(docs, start=1)]
    # sorted() is stable, so ties keep their retrieval order.
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
