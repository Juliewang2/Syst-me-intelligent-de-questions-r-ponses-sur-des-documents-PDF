"""
services/rag_service.py
--------------------------
The heart of the application: builds and runs the Retrieval-Augmented
Generation pipeline using LangChain Expression Language (LCEL).

Pipeline:
    question (+ chat history)
        -> [optional] condense into a standalone question
        -> retriever.invoke(question)                  (FAISS similarity search)
        -> format retrieved chunks into a context block
        -> RAG_PROMPT.format(context, chat_history, question)
        -> chat model from llm_factory (streaming or non-streaming)
        -> answer text

Every model is obtained from services/llm_factory.py with the
sensitivity of what it will read (services/sensitivity.py): the
condensation step sees the chat history, generation sees the history
and the retrieved chunks. Restricted input is answered by a local model.

Document summaries use the raw OpenAI SDK's Responses API with
structured (JSON-schema) output in openai mode, as before; in the other
modes (or for a restricted document) they go through
llm_factory.structured() instead.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import AsyncIterator, Callable, List, Optional, Union

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from openai import OpenAI
from pydantic import BaseModel

from config import get_settings
from services.llm_factory import get_chat_model, structured
from services.prompt_templates import (
    CONDENSE_QUESTION_PROMPT,
    RAG_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    build_summary_user_prompt,
)
from services.corrective_retriever import corrective_retrieve
from services.retriever import get_retriever
from services.sensitivity import (
    PUBLIC,
    context_sensitivity,
    document_sensitivity,
    history_sensitivity,
    stricter,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# Generation temperature (unchanged from the original single-model setup).
_ANSWER_TEMPERATURE = 0.2


@lru_cache
def _get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key or None,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_docs_for_context(docs: List[LCDocument]) -> str:
    """Render retrieved chunks into a numbered, citation-friendly block."""
    if not docs:
        return "(No relevant excerpts were found in the uploaded document(s).)"

    parts = []
    for i, doc in enumerate(docs, start=1):
        header = " | ".join([f"Source {i}", *_citation_fields(doc), _page_label(doc)])
        parts.append(f"[{header}]\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def _citation_fields(doc: LCDocument) -> List[str]:
    """Title (or file name), then UN symbol and date when the chunk has
    corpus metadata. Without it this is just the file name, as before."""
    meta = doc.metadata
    fields = [meta.get("title") or meta.get("document_name", "document")]
    fields += [meta[key] for key in ("un_symbol", "date") if meta.get(key)]
    return fields


def _page_label(doc: LCDocument) -> str:
    page = doc.metadata.get("page", "?")
    page_end = doc.metadata.get("page_end", page)
    return f"page {page}" if page_end == page else f"pages {page}-{page_end}"


def docs_to_source_dicts(docs: List[LCDocument]) -> List[dict]:
    """Convert retrieved LangChain Documents into serializable source
    citations for the API response."""
    sources = []
    for doc in docs:
        snippet = doc.page_content.strip()
        if len(snippet) > 280:
            snippet = snippet[:280].rsplit(" ", 1)[0] + "..."
        sources.append(
            {
                "document_id": doc.metadata.get("document_id"),
                "document_name": doc.metadata.get("document_name", "document"),
                "page": doc.metadata.get("page"),
                "page_end": doc.metadata.get("page_end", doc.metadata.get("page")),
                "chunk_index": doc.metadata.get("chunk_index"),
                "text_snippet": snippet,
                "rerank_score": doc.metadata.get("rerank_score"),
                "vector_similarity": doc.metadata.get("vector_similarity"),
                "title": doc.metadata.get("title"),
                "un_symbol": doc.metadata.get("un_symbol"),
                "date": doc.metadata.get("date"),
                "classification": doc.metadata.get("classification"),
            }
        )
    return sources


# ---------------------------------------------------------------------------
# Question condensation (turns a follow-up into a standalone query)
# ---------------------------------------------------------------------------

def _condense_question(question: str, chat_history: List[BaseMessage]) -> str:
    if not chat_history:
        return question

    llm = get_chat_model("rewrite", history_sensitivity(chat_history), temperature=0.0)
    chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    try:
        rewritten = chain.invoke({"question": question, "chat_history": chat_history})
        return rewritten.strip() or question
    except Exception as exc:  # noqa: BLE001
        logger.warning("Question condensation failed, falling back to raw question: %s", exc)
        return question


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    standalone_question: str
    documents: List[LCDocument]
    # Corrective retrieval summary (action, attempts, rewrites...); None
    # when corrective retrieval is off.
    trace: Optional[dict] = None


def retrieve_context(
    question: str,
    chat_history: List[BaseMessage],
    owner_id: str,
    document_id: Optional[str] = None,
    k: Optional[int] = None,
    hybrid: Optional[bool] = None,
    use_rerank: Optional[bool] = None,
    on_status: Optional[Callable[[dict], None]] = None,
) -> RetrievalResult:
    """Condense the question, then retrieve - through the corrective loop
    when CORRECTIVE_RETRIEVAL_ENABLED (which calls `on_status` before
    searching again)."""
    standalone_question = _condense_question(question, chat_history)
    if settings.corrective_retrieval_enabled:
        corrected = corrective_retrieve(
            owner_id, standalone_question, document_id=document_id, k=k,
            hybrid=hybrid, use_rerank=use_rerank, on_status=on_status,
        )
        return RetrievalResult(standalone_question, corrected.documents, corrected.trace())
    retriever = get_retriever(
        owner_id, document_id=document_id, k=k, hybrid=hybrid, use_rerank=use_rerank
    )
    documents = retriever.invoke(standalone_question)
    return RetrievalResult(standalone_question=standalone_question, documents=documents)


# ---------------------------------------------------------------------------
# Answer generation (non-streaming)
# ---------------------------------------------------------------------------

def _answer_model(chat_history: List[BaseMessage], documents: List[LCDocument], streaming: bool):
    """The generation model, chosen for the most restricted of the
    history and the retrieved chunks."""
    sensitivity = stricter(context_sensitivity(documents), history_sensitivity(chat_history))
    return get_chat_model("generate", sensitivity, streaming=streaming, temperature=_ANSWER_TEMPERATURE)


def answer_from_documents(
    question: str, chat_history: List[BaseMessage], documents: List[LCDocument]
) -> str:
    """The "generation" half of RAG: answer from already-retrieved chunks."""
    llm = _answer_model(chat_history, documents, streaming=False)
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {
            "context": format_docs_for_context(documents),
            "question": question,
            "chat_history": chat_history,
        }
    )


def generate_answer_traced(
    question: str,
    chat_history: List[BaseMessage],
    owner_id: str,
    document_id: Optional[str] = None,
) -> tuple[str, List[dict], Optional[dict]]:
    """Run the full RAG pipeline and return (answer_text, sources,
    retrieval trace or None)."""
    retrieval = retrieve_context(question, chat_history, owner_id, document_id=document_id)
    answer = answer_from_documents(question, chat_history, retrieval.documents)
    return answer, docs_to_source_dicts(retrieval.documents), retrieval.trace


def generate_answer(
    question: str,
    chat_history: List[BaseMessage],
    owner_id: str,
    document_id: Optional[str] = None,
) -> tuple[str, List[dict]]:
    """Run the full RAG pipeline and return (answer_text, sources)."""
    answer, sources, _ = generate_answer_traced(question, chat_history, owner_id, document_id)
    return answer, sources


async def _retrieve_with_status(
    question: str, chat_history: List[BaseMessage], owner_id: str, document_id: Optional[str]
) -> AsyncIterator[Union[dict, RetrievalResult]]:
    """Run retrieval in a worker thread and yield its status events as
    they happen, then the RetrievalResult.

    The corrective loop runs in that thread and reports "searching
    again" through on_status. loop.call_soon_threadsafe hands each status
    to an asyncio.Queue on the event loop - the only thread-safe way in -
    and this generator waits on whichever comes first, a status or the
    end of retrieval, so statuses reach the client immediately without
    polling. The thread's result is also delivered through the loop, so
    every status scheduled before it is already queued when it ends."""
    loop = asyncio.get_running_loop()
    statuses: asyncio.Queue = asyncio.Queue()

    def on_status(status: dict) -> None:
        loop.call_soon_threadsafe(statuses.put_nowait, status)

    task = asyncio.ensure_future(
        asyncio.to_thread(
            retrieve_context, question, chat_history, owner_id, document_id=document_id, on_status=on_status
        )
    )
    while not task.done():
        next_status = asyncio.ensure_future(statuses.get())
        done, _ = await asyncio.wait({task, next_status}, return_when=asyncio.FIRST_COMPLETED)
        if next_status in done:
            yield {"type": "status", **next_status.result()}
        else:
            next_status.cancel()
    while not statuses.empty():
        yield {"type": "status", **statuses.get_nowait()}
    yield task.result()


# ---------------------------------------------------------------------------
# Answer generation (streaming, for Server-Sent Events)
# ---------------------------------------------------------------------------

async def generate_answer_stream(
    question: str,
    chat_history: List[BaseMessage],
    owner_id: str,
    document_id: Optional[str] = None,
) -> AsyncIterator[dict]:
    """
    Async generator that first yields any 'status' events of corrective
    retrieval ("searching again..."), then a single 'sources' event, then
    a sequence of 'token' events as the LLM streams its answer, and
    finally a 'done' event (with the retrieval trace, None when corrective
    retrieval is off). Designed to be consumed directly by an SSE
    endpoint.
    """
    # Retrieval makes blocking network calls (embedding, rewrite, rerank);
    # it runs in a worker thread so the event loop keeps serving others,
    # and its "searching again" statuses are streamed before the sources.
    retrieval: Optional[RetrievalResult] = None
    async for item in _retrieve_with_status(question, chat_history, owner_id, document_id):
        if isinstance(item, RetrievalResult):
            retrieval = item
        else:
            yield item
    sources = docs_to_source_dicts(retrieval.documents)

    yield {"type": "sources", "sources": sources}

    context = format_docs_for_context(retrieval.documents)
    llm = _answer_model(chat_history, retrieval.documents, streaming=True)
    chain = RAG_PROMPT | llm | StrOutputParser()

    full_answer_parts: List[str] = []
    try:
        async for token in chain.astream(
            {
                "context": context,
                "question": question,
                "chat_history": chat_history,
            }
        ):
            if token:
                full_answer_parts.append(token)
                yield {"type": "token", "content": token}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Streaming generation failed")
        yield {"type": "error", "message": f"Generation failed: {exc}"}
        return

    yield {
        "type": "done",
        "full_answer": "".join(full_answer_parts),
        "sources": sources,
        "retrieval_trace": retrieval.trace,
    }


# ---------------------------------------------------------------------------
# Structured output via the raw OpenAI Responses API
# ---------------------------------------------------------------------------

SUMMARY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A concise, descriptive title for the document."},
        "overview": {
            "type": "string",
            "description": "A 2-4 sentence overview of what the document is about.",
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The most important concrete takeaways, ordered by importance.",
        },
        "document_type": {
            "type": "string",
            "description": "A short label for the kind of document, e.g. 'research paper', "
            "'legal contract', 'financial report', 'manual', 'resume'.",
        },
        "estimated_reading_time_minutes": {
            "type": "integer",
            "description": "Estimated minutes an average reader would need to read the full document.",
        },
    },
    "required": [
        "title",
        "overview",
        "key_points",
        "document_type",
        "estimated_reading_time_minutes",
    ],
    "additionalProperties": False,
}


class DocumentSummary(BaseModel):
    """Same fields as SUMMARY_JSON_SCHEMA, for llm_factory.structured()."""

    title: str
    overview: str
    key_points: List[str]
    document_type: str
    estimated_reading_time_minutes: int


def generate_structured_summary(
    document_text: str, max_key_points: int = 5, classification: Optional[str] = None
) -> dict:
    """
    A strongly-typed summary of a document. In openai mode, for a
    document that may go to OpenAI, this calls the OpenAI Responses
    API's structured output (JSON Schema) directly, as before; otherwise
    it uses the factory's model for the document's sensitivity.
    """
    # Very large documents are truncated to keep latency/cost predictable.
    truncated_text = document_text[:60_000]
    sensitivity = document_sensitivity(classification)
    if settings.deployment_mode != "openai" or sensitivity != PUBLIC:
        return _summary_via_factory(truncated_text, max_key_points, sensitivity)

    client = _get_openai_client()

    response = client.responses.create(
        model=settings.openai_chat_model,
        input=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": build_summary_user_prompt(truncated_text, max_key_points)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "document_summary",
                "schema": SUMMARY_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    import json

    return json.loads(response.output_text)


def _summary_via_factory(text: str, max_key_points: int, sensitivity: str) -> dict:
    llm = get_chat_model("generate", sensitivity, temperature=0.0)
    summary: DocumentSummary = structured(llm, DocumentSummary).invoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=build_summary_user_prompt(text, max_key_points)),
        ]
    )
    return summary.model_dump()
