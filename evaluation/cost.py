"""
evaluation/cost.py
--------------------
Counts the tokens and dollars the system under test spends per question.

Token counts come from LangChain itself: every chat model call ends with
an AIMessage whose `usage_metadata` holds input/output tokens. A callback
handler collects them. It is attached through LangChain's configure hook
(a context variable), so every model call made inside `track_usage()` is
seen - including the reranker's and the answer chain's - without adding
a callbacks parameter to the services. The judge is called outside that
block, so its tokens are never billed to the system being measured.

Dollars = tokens x the per-million price in evaluation/prices.json,
looked up by provider + model. Providers report dated model names
(gpt-4o-mini-2024-07-18), so a trailing date is stripped if the exact
name is unknown. Unknown models give cost None and a warning rather
than a made-up number. Embedding calls are not chat model calls and are
not counted.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook

logger = logging.getLogger(__name__)

PRICES_FILE = Path(__file__).resolve().parent / "prices.json"
_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")
_warned: set = set()


class UsageTracker(BaseCallbackHandler):
    """Collects one record per chat model call:
    {provider, model, input_tokens, output_tokens}."""

    def __init__(self) -> None:
        self.calls: List[dict] = []
        self._providers: Dict[UUID, str] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        provider = (kwargs.get("metadata") or {}).get("ls_provider")
        if not provider:
            # e.g. ["langchain", "chat_models", "openai", "ChatOpenAI"]
            ids = (serialized or {}).get("id") or []
            provider = ids[2] if len(ids) > 2 else "unknown"
        with self._lock:
            self._providers[run_id] = provider

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        message = _first_message(response)
        usage = getattr(message, "usage_metadata", None) or {}
        if not usage:
            logger.warning("Model call returned no usage_metadata; counting 0 tokens")
        model = (getattr(message, "response_metadata", None) or {}).get("model_name") or (
            response.llm_output or {}
        ).get("model_name", "unknown")
        with self._lock:
            provider = self._providers.pop(run_id, "unknown")
            self.calls.append(
                {
                    "provider": provider,
                    "model": model,
                    "input_tokens": int(usage.get("input_tokens", 0)),
                    "output_tokens": int(usage.get("output_tokens", 0)),
                }
            )


def _first_message(response: LLMResult) -> Any:
    try:
        return response.generations[0][0].message
    except (IndexError, AttributeError):
        return None


_current_tracker: ContextVar[Optional[UsageTracker]] = ContextVar("eval_usage_tracker", default=None)
register_configure_hook(_current_tracker, inheritable=True)


@contextmanager
def track_usage() -> Iterator[UsageTracker]:
    """Record every chat model call made inside the block."""
    tracker = UsageTracker()
    token = _current_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _current_tracker.reset(token)


def load_prices(path: Path = PRICES_FILE) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _price_entry(provider: str, model: str, prices: dict) -> Optional[dict]:
    table = prices.get(provider, {})
    return table.get(model) or table.get(_DATE_SUFFIX.sub("", model))


def call_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int, prices: dict
) -> Optional[float]:
    """Dollar cost of one call, or None if the model has no known price."""
    entry = _price_entry(provider, model, prices)
    if entry is None:
        if (provider, model) not in _warned:
            _warned.add((provider, model))
            logger.warning("No price for %s/%s in prices.json; cost recorded as null", provider, model)
        return None
    return (input_tokens * entry["input_per_1m"] + output_tokens * entry["output_per_1m"]) / 1_000_000


def usage_summary(calls: List[dict], prices: dict) -> dict:
    """Per-question totals: tokens, call count and cost (None if any
    call's price is unknown, so a partial sum is never shown as total)."""
    costs = [call_cost(c["provider"], c["model"], c["input_tokens"], c["output_tokens"], prices) for c in calls]
    return {
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "llm_calls": len(calls),
        "cost_usd": None if any(c is None for c in costs) else round(sum(costs), 8),
    }
