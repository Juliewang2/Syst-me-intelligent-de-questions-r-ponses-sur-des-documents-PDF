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
(structured output). Candidates below `RERANK_MIN_SCORE` are dropped,
which also acts as a relevance threshold: if nothing is relevant, the
answer step is told so instead of being fed noise.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

from langchain_core.documents import Document as LCDocument
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import get_settings
from services.prompt_templates import RERANK_PROMPT

logger = logging.getLogger(__name__)
settings = get_settings()

# Long chunks are shortened in the rerank prompt to bound its cost.
_MAX_PASSAGE_CHARS = 1200


class _PassageScore(BaseModel):
    index: int = Field(description="The passage number, as shown in [Passage N].")
    score: int = Field(description="Relevance from 0 (unrelated) to 10 (directly answers it).")


class _RerankResult(BaseModel):
    scores: List[_PassageScore]


@lru_cache
def _get_rerank_chain():
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key or None,
        temperature=0.0,
        timeout=settings.rerank_timeout_seconds,
        max_retries=0,
    )
    return RERANK_PROMPT | llm.with_structured_output(_RerankResult, method="json_schema", strict=True)


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
        result: _RerankResult = _get_rerank_chain().invoke(
            {"question": question, "passages": _format_passages(docs)}
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranking failed, keeping retrieval order: %s", exc)
        return None

    scores = {s.index: max(0, min(10, s.score)) for s in result.scores}
    scored = [(doc, scores.get(i, 0)) for i, doc in enumerate(docs, start=1)]
    # sorted() is stable, so ties keep their retrieval order.
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
