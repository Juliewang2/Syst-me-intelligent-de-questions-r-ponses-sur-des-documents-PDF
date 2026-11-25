"""
services/llm_factory.py
-------------------------
The one place chat models and embeddings are created, so the same code
can run against OpenAI, Mistral or local models (Ollama) by changing
DEPLOYMENT_MODE - and so restricted text can be kept away from external
providers in a single, testable spot.

Model choice for a call = deployment mode's default provider, then the
role's override (GENERATE_PROVIDER, RERANK_MODEL, ...), then the
sensitivity rule: when the context is DR_SIMULE and that provider is not
in DR_ALLOWED_PROVIDERS, the call is rerouted to Ollama. Callers compute
the sensitivity (services/sensitivity.py) and pass it in; they never
pick a provider themselves.

Structured output differs by provider, so `structured()` hides it:
  openai  - json_schema, strict (the API guarantees the schema)
  mistral - function_calling (langchain-mistralai 0.2.4, the newest
            release compatible with langchain-core 0.3, has no
            json_schema method)
  ollama  - json_schema (passed as Ollama's `format`), temperature 0,
            and the schema repeated in the system prompt: small local
            models follow it more reliably when they can read it.
A parse failure is retried once, then raised as StructuredOutputError;
every caller already degrades on errors (rerank keeps the fused order).

Each model request is recorded (role, provider, model, sensitivity,
rerouted) in a context-local call log, read by evaluation/run_eval.py
and traces; only metadata is logged, never prompt text.
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

Provider = Literal["openai", "mistral", "ollama"]
Sensitivity = Literal["PUBLIC", "DR_SIMULE"]
Role = Literal["generate", "rerank", "rewrite", "verify", "agent"]

PROVIDERS: Tuple[str, ...] = ("openai", "mistral", "ollama")
ROLES: Tuple[str, ...] = ("generate", "rerank", "rewrite", "verify", "agent")
LOCAL_PROVIDER = "ollama"

# Chat provider each deployment mode uses when a role sets none.
_MODE_CHAT_PROVIDER: Dict[str, str] = {
    "openai": "openai",
    "mistral": "mistral",
    "souverain": "ollama",
    "hybride": "openai",
}


class StructuredOutputError(Exception):
    """The model's output could not be parsed into the schema, even after
    one retry."""


# ---------------------------------------------------------------------------
# Call log
# ---------------------------------------------------------------------------

_call_log: ContextVar[Optional[List[dict]]] = ContextVar("llm_call_log", default=None)


@contextmanager
def llm_call_log() -> Iterator[List[dict]]:
    """Collect a record of every model request made inside the block.
    The list is shared with worker threads started from it
    (asyncio.to_thread copies the context, not the list)."""
    records: List[dict] = []
    token = _call_log.set(records)
    try:
        yield records
    finally:
        _call_log.reset(token)


def record_event(event: str, **fields: Any) -> None:
    """Append an entry to the active call log (no-op outside one)."""
    records = _call_log.get()
    if records is not None:
        records.append({"event": event, **fields})


# ---------------------------------------------------------------------------
# Choosing a provider and model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelChoice:
    role: str
    provider: str
    model: str
    sensitivity: str
    rerouted: bool


def _check_provider(provider: str) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown model provider {provider!r} (expected one of {', '.join(PROVIDERS)})")
    return provider


def _default_chat_model(provider: str) -> str:
    return {
        "openai": settings.openai_chat_model,
        "mistral": settings.mistral_chat_model,
        "ollama": settings.local_chat_model,
    }[provider]


def resolve_chat_model(role: str, sensitivity: str = "PUBLIC") -> ModelChoice:
    """Which provider and model serve `role` for a context of this
    sensitivity (no model is created)."""
    if role not in ROLES:
        raise ValueError(f"Unknown model role {role!r} (expected one of {', '.join(ROLES)})")
    provider = _check_provider(
        getattr(settings, f"{role}_provider") or _MODE_CHAT_PROVIDER[settings.deployment_mode]
    )
    model = getattr(settings, f"{role}_model")
    rerouted = False
    if sensitivity == "DR_SIMULE" and provider not in settings.dr_allowed_providers:
        # The role's model override names an external model; use the
        # configured local one instead.
        provider, model, rerouted = LOCAL_PROVIDER, "", True
    return ModelChoice(role, provider, model or _default_chat_model(provider), sensitivity, rerouted)


# ---------------------------------------------------------------------------
# Building chat models
# ---------------------------------------------------------------------------

def _timeout(provider: str, role: str) -> Tuple[float, int]:
    """(timeout seconds, retries). Reranking is optional, so on hosted
    providers it keeps its short timeout and no retries, as before."""
    if provider == LOCAL_PROVIDER:
        return settings.local_timeout_seconds, 0
    if role == "rerank":
        return settings.rerank_timeout_seconds, 0
    return settings.openai_timeout_seconds, settings.openai_max_retries


def _openai(model: str, temperature: Optional[float], streaming: bool, timeout: float, retries: int) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    kwargs = {} if temperature is None else {"temperature": temperature}
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key or None,
        streaming=streaming,
        timeout=timeout,
        max_retries=retries,
        **kwargs,
    )


def _mistral(model: str, temperature: Optional[float], streaming: bool, timeout: float, retries: int) -> BaseChatModel:
    from langchain_mistralai import ChatMistralAI

    kwargs = {} if temperature is None else {"temperature": temperature}
    return ChatMistralAI(
        model=model,
        mistral_api_key=settings.mistral_api_key or "missing-key",
        streaming=streaming,
        timeout=math.ceil(timeout),  # the client takes whole seconds
        max_retries=retries,
        **kwargs,
    )


def _ollama(model: str, temperature: Optional[float], streaming: bool, timeout: float, retries: int) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    kwargs = {} if temperature is None else {"temperature": temperature}
    # ChatOllama streams through astream() whatever `streaming` says.
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": timeout},
        **kwargs,
    )


# Provider -> constructor. Tests replace entries with fake models.
CHAT_BUILDERS: Dict[str, Callable[..., BaseChatModel]] = {
    "openai": _openai,
    "mistral": _mistral,
    "ollama": _ollama,
}

_model_cache: Dict[tuple, BaseChatModel] = {}


def get_chat_model(
    role: str,
    sensitivity: Sensitivity = "PUBLIC",
    streaming: bool = False,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """The chat model for `role`, given the sensitivity of the context it
    will see. Rerouted to Ollama when the context is restricted and the
    configured provider may not receive it."""
    choice = resolve_chat_model(role, sensitivity)
    timeout, retries = _timeout(choice.provider, role)
    key = (choice.provider, choice.model, streaming, temperature, timeout, retries)
    if key not in _model_cache:
        _model_cache[key] = CHAT_BUILDERS[choice.provider](choice.model, temperature, streaming, timeout, retries)

    record_event("model", **asdict(choice))
    logger.info(
        "LLM %s -> %s/%s (sensitivity=%s%s)",
        role, choice.provider, choice.model, sensitivity, ", rerouted_to_local" if choice.rerouted else "",
    )
    return _model_cache[key]


def provider_of(llm: BaseChatModel) -> str:
    """'openai' | 'mistral' | 'ollama' for a chat model instance."""
    params = llm._get_ls_params()  # noqa: SLF001 - LangChain's own provider tag
    provider = params.get("ls_provider", "")
    if provider in PROVIDERS:
        return provider
    raise ValueError(f"Cannot tell the provider of {type(llm).__name__} (ls_provider={provider!r})")


def reset_cache() -> None:
    """Forget built models and embeddings (tests, settings changes)."""
    _model_cache.clear()
    _embeddings.cache_clear()


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

def _schema_text(schema: Type[BaseModel]) -> str:
    return (
        "\n\nRespond only with a JSON object that matches this JSON schema:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


def _as_messages(value: Any) -> List[BaseMessage]:
    if isinstance(value, PromptValue):
        return value.to_messages()
    if isinstance(value, str):
        return [HumanMessage(content=value)]
    return list(value)


def _append_to_system(messages: List[BaseMessage], text: str) -> List[BaseMessage]:
    """Copy of `messages` with `text` added at the end of the system
    message (or as a new first system message)."""
    for i, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            updated = SystemMessage(content=f"{message.content}{text}")
            return [*messages[:i], updated, *messages[i + 1:]]
    return [SystemMessage(content=text.strip()), *messages]


def _bind_structured(llm: BaseChatModel, schema: Type[BaseModel]) -> Runnable:
    provider = provider_of(llm)
    if provider == "openai":
        return llm.with_structured_output(schema, method="json_schema", strict=True)
    if provider == "mistral":
        return llm.with_structured_output(schema, method="function_calling")
    local = llm.model_copy(update={"temperature": 0})
    add_schema = RunnableLambda(lambda value: _append_to_system(_as_messages(value), _schema_text(schema)))
    return add_schema | local.with_structured_output(schema, method="json_schema")


def _is_parse_error(exc: Exception) -> bool:
    from langchain_core.exceptions import OutputParserException
    from pydantic import ValidationError

    return isinstance(exc, (OutputParserException, ValidationError, json.JSONDecodeError))


def structured(llm: BaseChatModel, schema: Type[BaseModel]) -> Runnable:
    """A runnable returning `schema` instances from `llm`, using the
    provider's most reliable structured-output method. Unparseable output
    is retried once, then raises StructuredOutputError; other errors
    (timeouts, network) propagate unchanged."""
    bound = _bind_structured(llm, schema)
    provider = provider_of(llm)

    def invoke(value: Any, config: Optional[RunnableConfig] = None) -> BaseModel:
        record_event("structured_call", provider=provider, schema=schema.__name__)
        last_error: Optional[Exception] = None
        for _ in range(2):
            try:
                result = bound.invoke(value, config)
            except Exception as exc:  # noqa: BLE001
                if not _is_parse_error(exc):
                    raise
                last_error = exc
                continue
            if isinstance(result, schema):
                return result
            last_error = ValueError(f"model returned {type(result).__name__}, not {schema.__name__}")
        record_event("structured_failure", provider=provider, schema=schema.__name__)
        raise StructuredOutputError(f"{provider} output did not match {schema.__name__}: {last_error}")

    return RunnableLambda(invoke, name=f"structured[{schema.__name__}]")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class NormalizedEmbeddings(Embeddings):
    """Scales every vector to unit length. vector_store converts FAISS's
    squared L2 distance to cosine with `1 - d / 2`, which only holds for
    unit vectors; OpenAI's are, Ollama's are not guaranteed to be."""

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner

    @staticmethod
    def _unit(vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._unit(v) for v in self.inner.embed_documents(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self._unit(self.inner.embed_query(text))


def embedding_choice() -> Tuple[str, str]:
    """(provider, model) for embeddings. Hybrid and sovereign modes always
    embed locally: indexing sends every chunk, restricted ones included."""
    mode = settings.deployment_mode
    if mode in ("souverain", "hybride"):
        return LOCAL_PROVIDER, settings.local_embedding_model
    if mode == "mistral":
        return "mistral", settings.mistral_embedding_model
    return "openai", settings.openai_embedding_model


@lru_cache
def _embeddings(provider: str, model: str) -> Embeddings:
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            api_key=settings.openai_api_key or None,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
    if provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(
            model=model,
            mistral_api_key=settings.mistral_api_key or "missing-key",
            timeout=math.ceil(settings.openai_timeout_seconds),
            max_retries=settings.openai_max_retries,
        )
    from langchain_ollama import OllamaEmbeddings

    return NormalizedEmbeddings(
        OllamaEmbeddings(
            model=model,
            base_url=settings.ollama_base_url,
            client_kwargs={"timeout": settings.local_timeout_seconds},
        )
    )


def get_embeddings() -> Embeddings:
    """The embedding model for the current deployment mode (cached)."""
    return _embeddings(*embedding_choice())
