"""
services/query_rewriter.py
----------------------------
Reformulations of a question, for a second search when the first one
found little (services/corrective_retriever.py). This plays the part
that web search plays in CRAG: instead of looking elsewhere, look again
in the same corpus with better words.

Two failure causes are targeted:
  - wording: abbreviations and jargon ("CDN", "AGNU") that the documents
    spell out -> `expanded`, a formal version of the question;
  - language: a French question about English documents (or the
    reverse), where BM25 finds nothing in common -> `translations` into
    the other languages present in the corpus.

UN document symbols are normalized first, without any model call
("résolution 77-276" -> A/RES/77/276): exact symbols are what BM25
matches best, and this rewrite survives even if the model call fails.
The model then gets one call (role "rewrite", short timeout), with the
sensitivity of the passages it is shown, so restricted passages are only
ever reformulated by a local model. Failure or timeout -> no model
rewrites, never an exception.
"""

from __future__ import annotations

import logging
from typing import Collection, List, Optional, Sequence

from langchain_core.documents import Document as LCDocument
from pydantic import BaseModel, Field

from services.llm_factory import get_chat_model, structured
from services.prompt_templates import QUERY_REWRITE_PROMPT
from services.sensitivity import context_sensitivity
from services.un_votes import normalize_un_symbol
from services.vector_store import get_all_chunks

logger = logging.getLogger(__name__)

MAX_REWRITES = 3
MAX_TRANSLATIONS = 2
DEFAULT_MODES = frozenset({"expand", "translate"})
# Failed passages shown to the model: enough to see what was missed.
_CONTEXT_PASSAGES = 4
_CONTEXT_CHARS = 500


class QueryRewrites(BaseModel):
    expanded: str = Field(description="The question with abbreviations and jargon expanded, in formal terms.")
    translations: List[str] = Field(
        default_factory=list,
        description="The question translated into other corpus languages (at most 2).",
    )


def corpus_languages(owner_id: str) -> List[str]:
    """Languages of the owner's documents (from corpus metadata)."""
    return sorted({c.metadata["language"] for c in get_all_chunks(owner_id) if c.metadata.get("language")})


def symbol_rewrite(question: str) -> Optional[str]:
    """The question with the official UN symbol it mentions appended, or
    None when it mentions none or already uses the official form."""
    symbol = normalize_un_symbol(question)
    if symbol is None or symbol in question:
        return None
    return f"{question} ({symbol})"


def _translation_instruction(languages: Sequence[str], modes: Collection[str]) -> str:
    if "translate" not in modes or not languages:
        return "always return an empty list."
    return (
        f"the question translated into each of these document languages that differs from the "
        f"question's own language: {', '.join(languages)} (at most {MAX_TRANSLATIONS}). "
        f"Empty list if the question is already in the only language."
    )


def _format_context(docs: Optional[Sequence[LCDocument]]) -> str:
    if not docs:
        return "(none)"
    return "\n---\n".join(d.page_content.strip()[:_CONTEXT_CHARS] for d in docs[:_CONTEXT_PASSAGES])


def _model_rewrites(
    question: str,
    languages: Sequence[str],
    failed_context: Optional[Sequence[LCDocument]],
    modes: Collection[str],
) -> List[str]:
    llm = get_chat_model("rewrite", context_sensitivity(failed_context or []), temperature=0.0)
    try:
        result: QueryRewrites = (QUERY_REWRITE_PROMPT | structured(llm, QueryRewrites)).invoke(
            {
                "question": question,
                "failed_context": _format_context(failed_context),
                "translation_instruction": _translation_instruction(languages, modes),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query rewriting failed, no model rewrites: %s", exc)
        return []
    rewrites = [result.expanded] if "expand" in modes else []
    if "translate" in modes:
        rewrites += _clean(question, result.translations)[:MAX_TRANSLATIONS]
    return rewrites


def _clean(question: str, candidates: Sequence[Optional[str]]) -> List[str]:
    """Non-empty, distinct, different from the question, at most 3."""
    seen = {question.strip().casefold()}
    kept = []
    for text in candidates:
        text = (text or "").strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            kept.append(text)
    return kept[:MAX_REWRITES]


def rewrite_query(
    question: str,
    corpus_languages: Sequence[str],
    failed_context: Optional[Sequence[LCDocument]] = None,
    modes: Optional[Collection[str]] = None,
) -> List[str]:
    """Up to 3 reformulations of `question` (symbol-normalized first,
    then the model's), or [] if none could be produced."""
    modes = DEFAULT_MODES if modes is None else modes
    with_symbol = symbol_rewrite(question)
    model_rewrites = _model_rewrites(with_symbol or question, corpus_languages, failed_context, modes)
    return _clean(question, [with_symbol, *model_rewrites])
