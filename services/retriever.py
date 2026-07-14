"""
services/retriever.py
------------------------
Exposes the shared FAISS vector store as a LangChain `Retriever`
(via `VectorStoreRetriever.as_retriever`), optionally scoped to a
single document. This is the component the RAG chain calls to fetch
context for a user's question.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from config import get_settings
from services.vector_store import get_vectorstore, similarity_search

settings = get_settings()


class PDFChatRetriever(BaseRetriever):
    """A LangChain-compatible retriever backed by the shared FAISS
    vector store, with optional single-document scoping and a
    configurable top-k."""

    k: int = Field(default=settings.retriever_k)
    document_id: Optional[str] = Field(default=None)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[LCDocument]:  # noqa: ANN001
        return similarity_search(query, k=self.k, document_id=self.document_id)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[LCDocument]:  # noqa: ANN001
        # similarity_search is CPU/IO bound but fast for local FAISS;
        # a thin async wrapper keeps the interface consistent with the
        # rest of the async pipeline.
        return self._get_relevant_documents(query, run_manager=run_manager)


def get_retriever(document_id: Optional[str] = None, k: Optional[int] = None) -> PDFChatRetriever:
    """Factory for a scoped retriever instance."""
    return PDFChatRetriever(k=k or settings.retriever_k, document_id=document_id)


def has_indexed_documents() -> bool:
    """Whether the vector store currently has anything to search."""
    vs = get_vectorstore(create_if_missing=False)
    return vs is not None and len(vs.docstore._dict) > 0  # noqa: SLF001
