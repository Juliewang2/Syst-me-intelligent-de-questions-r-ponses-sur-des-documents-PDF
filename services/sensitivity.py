"""
services/sensitivity.py
-------------------------
How restricted is the text a model call is about to see?

Every LLM call (answer generation, rerank, question condensation,
summary, and later rewrite / verify / agent) first computes the
sensitivity of its input here and passes it to llm_factory, which
reroutes restricted input to a local model. Keeping the rule in one
module means "may this text leave the machine?" is answered the same
way everywhere.

Rules:
  - A chunk (or a source dict saved with a message) is restricted when
    its classification is DR_SIMULE.
  - A chunk without a classification is restricted too - the same safe
    default as services/document_meta.py. The one exception is the
    legacy setup (openai mode, corpus metadata off): there nothing is
    classified and every call has always gone to OpenAI, so unlabeled
    chunks count as PUBLIC and the default behaviour stays unchanged.
  - Chat history is restricted when any earlier answer in it was built
    from restricted sources: an answer can repeat restricted text, and
    history is sent along with the next question.
  - A request is as restricted as the most restricted of its parts.

Ingestion has a matching rule: a DR_SIMULE document is refused when the
embedding provider may not receive restricted text, since indexing
sends every chunk to it (`check_ingestion_allowed`).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import BaseMessage

from config import get_settings
from services.llm_factory import Sensitivity, embedding_choice

settings = get_settings()

PUBLIC: Sensitivity = "PUBLIC"
RESTRICTED: Sensitivity = "DR_SIMULE"

# Key under which load_chat_history marks a restricted earlier answer
# (in AIMessage.response_metadata, which is never sent to a provider).
HISTORY_KEY = "sensitivity"


class ClassificationNotAllowed(Exception):
    """A restricted document cannot be indexed in this deployment mode."""


def _unlabeled_is_public() -> bool:
    return settings.deployment_mode == "openai" and not settings.corpus_metadata_enabled


def is_restricted(metadata: Mapping) -> bool:
    """Is a chunk's metadata (or a saved source dict) restricted?"""
    classification = metadata.get("classification")
    if classification is None:
        return not _unlabeled_is_public()
    return classification != PUBLIC


def stricter(*levels: Optional[str]) -> Sensitivity:
    """DR_SIMULE if any level is DR_SIMULE, else PUBLIC."""
    return RESTRICTED if RESTRICTED in levels else PUBLIC


def context_sensitivity(docs: Iterable[LCDocument]) -> Sensitivity:
    """DR_SIMULE if any chunk is restricted, else PUBLIC."""
    return RESTRICTED if any(is_restricted(d.metadata) for d in docs) else PUBLIC


def sources_sensitivity(sources: Optional[Sequence[Mapping]]) -> Sensitivity:
    """Same rule for the source dicts saved with an assistant message."""
    return RESTRICTED if any(is_restricted(s) for s in sources or []) else PUBLIC


def history_sensitivity(history: Sequence[BaseMessage]) -> Sensitivity:
    """DR_SIMULE if any message of the history is marked restricted."""
    levels = [getattr(m, "response_metadata", {}).get(HISTORY_KEY) for m in history]
    return stricter(*levels)


def document_sensitivity(classification: Optional[str]) -> Sensitivity:
    """Sensitivity of a whole document from its stored classification."""
    return RESTRICTED if is_restricted({"classification": classification}) else PUBLIC


def check_ingestion_allowed(classification: Optional[str]) -> None:
    """Raise ClassificationNotAllowed if a document of this classification
    may not be embedded by the current embedding provider."""
    if document_sensitivity(classification) == PUBLIC:
        return
    provider, model = embedding_choice()
    if provider not in settings.dr_allowed_providers:
        raise ClassificationNotAllowed(
            f"Restricted (DR_SIMULE) documents cannot be imported in deployment mode "
            f"'{settings.deployment_mode}': indexing would send their text to {provider} "
            f"({model}). Use DEPLOYMENT_MODE=hybride or souverain, or mark the document "
            f"PUBLIC if it really is public."
        )
