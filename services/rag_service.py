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
        -> ChatOpenAI (streaming or non-streaming)
        -> answer text

Also demonstrates the raw OpenAI Python SDK's Responses API with
structured (JSON-schema) output for document summarization — a
feature that sits alongside, rather than inside, the LangChain chain.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import AsyncIterator, List, Optional

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import OpenAI

from config import get_settings
from services.prompt_templates import (
    CONDENSE_QUESTION_PROMPT,
    RAG_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    build_summary_user_prompt,
)
from services.retriever import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# LLM factories
# ---------------------------------------------------------------------------

@lru_cache
def _get_chat_llm(streaming: bool, temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key or None,
        temperature=temperature,
        streaming=streaming,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )


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
        name = doc.metadata.get("document_name", "document")
        parts.append(f"[Source {i} | {name} | {_page_label(doc)}]\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


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
            }
        )
    return sources


# ---------------------------------------------------------------------------
# Question condensation (turns a follow-up into a standalone query)
# ---------------------------------------------------------------------------

def _condense_question(question: str, chat_history: List[BaseMessage]) -> str:
    if not chat_history:
        return question

    llm = _get_chat_llm(streaming=False, temperature=0.0)
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


def retrieve_context(
    question: str,
    chat_history: List[BaseMessage],
    owner_id: str,
    document_id: Optional[str] = None,
    k: Optional[int] = None,
    hybrid: Optional[bool] = None,
    use_rerank: Optional[bool] = None,
) -> RetrievalResult:
    standalone_question = _condense_question(question, chat_history)
    retriever = get_retriever(
        owner_id, document_id=document_id, k=k, hybrid=hybrid, use_rerank=use_rerank
    )
    documents = retriever.invoke(standalone_question)
    return RetrievalResult(standalone_question=standalone_question, documents=documents)


# ---------------------------------------------------------------------------
# Answer generation (non-streaming)
# ---------------------------------------------------------------------------

def answer_from_documents(
    question: str, chat_history: List[BaseMessage], documents: List[LCDocument]
) -> str:
    """The "generation" half of RAG: answer from already-retrieved chunks."""
    llm = _get_chat_llm(streaming=False)
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {
            "context": format_docs_for_context(documents),
            "question": question,
            "chat_history": chat_history,
        }
    )


def generate_answer(
    question: str,
    chat_history: List[BaseMessage],
    owner_id: str,
    document_id: Optional[str] = None,
) -> tuple[str, List[dict]]:
    """Run the full RAG pipeline and return (answer_text, sources)."""
    retrieval = retrieve_context(question, chat_history, owner_id, document_id=document_id)
    answer = answer_from_documents(question, chat_history, retrieval.documents)
    return answer, docs_to_source_dicts(retrieval.documents)


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
    Async generator that first yields a single 'sources' event, then a
    sequence of 'token' events as the LLM streams its answer, and
    finally a 'done' event. Designed to be consumed directly by an SSE
    endpoint.
    """
    # Retrieval makes blocking network calls (embedding, rewrite, rerank);
    # run it in a worker thread so the event loop keeps serving others.
    retrieval = await asyncio.to_thread(
        retrieve_context, question, chat_history, owner_id, document_id=document_id
    )
    sources = docs_to_source_dicts(retrieval.documents)

    yield {"type": "sources", "sources": sources}

    context = format_docs_for_context(retrieval.documents)
    llm = _get_chat_llm(streaming=True)
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

    yield {"type": "done", "full_answer": "".join(full_answer_parts), "sources": sources}


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


def generate_structured_summary(document_text: str, max_key_points: int = 5) -> dict:
    """
    Uses the OpenAI Responses API's structured output (JSON Schema)
    feature to produce a strongly-typed summary of a document. This
    calls the OpenAI SDK directly (rather than going through
    LangChain) to showcase the native Responses API.
    """
    client = _get_openai_client()

    # The Responses API caps input size generously, but we defensively
    # truncate very large documents to keep latency/cost predictable.
    truncated_text = document_text[:60_000]

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
