"""
services/vector_store.py
--------------------------
Manages a single, persistent FAISS vector store shared across all
uploaded documents. Chunks are tagged with `document_id` metadata so
that retrieval can optionally be scoped to one document, while still
allowing "search across everything I've uploaded" style queries.

The index is persisted to disk (`vector_store_dir`) so it survives
application restarts, and is protected by a lock to avoid concurrent
read/modify/write races when multiple requests touch it at once.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument

from config import get_settings
from services.embedding_service import get_embeddings

logger = logging.getLogger(__name__)
settings = get_settings()

_INDEX_NAME = "faiss_index"
_lock = threading.RLock()

_vectorstore_cache: Optional[FAISS] = None


class VectorStoreError(Exception):
    """Raised for unrecoverable vector store operations."""


def _index_path() -> Path:
    return settings.vector_store_dir


def _index_exists() -> bool:
    path = _index_path()
    return (path / f"{_INDEX_NAME}.faiss").exists() and (path / f"{_INDEX_NAME}.pkl").exists()


def _load_from_disk() -> Optional[FAISS]:
    if not _index_exists():
        return None
    try:
        return FAISS.load_local(
            str(_index_path()),
            get_embeddings(),
            index_name=_INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load FAISS index from disk: %s", exc)
        return None


def _save_to_disk(vs: FAISS) -> None:
    vs.save_local(str(_index_path()), index_name=_INDEX_NAME)


def get_vectorstore(create_if_missing: bool = False) -> Optional[FAISS]:
    """Return the process-wide FAISS vector store, loading it from disk
    on first access. Returns None if no index exists yet and
    `create_if_missing` is False."""
    global _vectorstore_cache

    with _lock:
        if _vectorstore_cache is not None:
            return _vectorstore_cache

        vs = _load_from_disk()
        if vs is None and create_if_missing:
            # Bootstrap an empty index with a throwaway document, then
            # immediately remove it, so we have a valid FAISS structure
            # to append to later.
            placeholder = LCDocument(
                page_content="pdf-chat vector store initialized",
                metadata={"document_id": "__init__", "bootstrap": True},
            )
            vs = FAISS.from_documents([placeholder], get_embeddings())
            ids_to_remove = [
                doc_id
                for doc_id, doc in vs.docstore._dict.items()  # noqa: SLF001
                if doc.metadata.get("bootstrap")
            ]
            if ids_to_remove:
                vs.delete(ids_to_remove)
            _save_to_disk(vs)

        _vectorstore_cache = vs
        return _vectorstore_cache


def add_document_chunks(chunks: List[LCDocument]) -> List[str]:
    """Embed and add a batch of chunks (typically all chunks belonging
    to one uploaded document) to the shared vector store, persisting
    the updated index to disk. Returns the generated chunk ids."""
    if not chunks:
        return []

    with _lock:
        vs = get_vectorstore(create_if_missing=True)
        if vs is None:
            raise VectorStoreError("Vector store could not be initialized.")

        ids = vs.add_documents(chunks)
        _save_to_disk(vs)
        return ids


def delete_document(document_id: str) -> int:
    """Remove all chunks belonging to `document_id` from the vector
    store. Returns the number of chunks removed."""
    with _lock:
        vs = get_vectorstore(create_if_missing=False)
        if vs is None:
            return 0

        ids_to_remove = [
            doc_id
            for doc_id, doc in vs.docstore._dict.items()  # noqa: SLF001
            if doc.metadata.get("document_id") == document_id
        ]
        if not ids_to_remove:
            return 0

        vs.delete(ids_to_remove)
        _save_to_disk(vs)
        return len(ids_to_remove)


def similarity_search(
    query: str,
    k: int = 4,
    document_id: Optional[str] = None,
) -> List[LCDocument]:
    """Run a similarity search against the shared vector store,
    optionally filtered to a single document."""
    vs = get_vectorstore(create_if_missing=False)
    if vs is None:
        return []

    search_kwargs = {}
    if document_id:
        search_kwargs["filter"] = {"document_id": document_id}

    # Over-fetch a little when filtering, since FAISS applies the
    # metadata filter after the nearest-neighbor search.
    fetch_k = k * 4 if document_id else k
    results = vs.similarity_search(query, k=fetch_k, **search_kwargs)
    return results[:k]


def reset_cache() -> None:
    """Clear the in-memory cache, forcing a reload from disk on next
    access. Mainly useful for tests."""
    global _vectorstore_cache
    with _lock:
        _vectorstore_cache = None
