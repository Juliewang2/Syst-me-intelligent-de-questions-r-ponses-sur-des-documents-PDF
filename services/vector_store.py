"""
services/vector_store.py
--------------------------
Manages persistent FAISS vector stores - one per user. Each user's
chunks live in their own index under `vector_store_dir/<owner_id>/`,
so one user's search can never return another user's documents, and
a search only has to scan that user's vectors.

Within a user's index, chunks are tagged with `document_id` metadata
so retrieval can be scoped to a single document (and, with corpus
metadata, to source / type / classification / dates - see
services/chunk_filters.py). LangChain's FAISS
wrapper applies metadata filters *after* the nearest-neighbour search,
so when filtering we search the whole index (`fetch_k = ntotal`) to be
sure matching chunks are never cut off before the filter runs. For
the default flat (brute-force) index this costs nothing extra: every
vector is compared to the query either way.

Indexes are persisted to disk so they survive restarts, and a lock
avoids concurrent read/modify/write races.

Vectors from different embedding models are not comparable, so each
model has its own directory, vector_store_dir/<provider>__<model>/
<owner_id>/, with an index_meta.json recording the model. Loading an
index whose metadata names another model, or finding a user's index
only under another model's directory (e.g. after switching
DEPLOYMENT_MODE), raises IndexModelMismatch instead of silently
searching the wrong vector space or an empty one; scripts/reindex.py
rebuilds the index for the current model. An index from before this
layout (vector_store_dir/<owner_id>/) was built with OpenAI embeddings;
in openai mode it is moved into the new layout on first access, so
existing data keeps working unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument

from config import get_settings
from services.chunk_filters import make_predicate
from services.embedding_service import get_embeddings
from services.llm_factory import embedding_choice

logger = logging.getLogger(__name__)
settings = get_settings()

_INDEX_NAME = "faiss_index"
_META_NAME = "index_meta.json"
_lock = threading.RLock()

# Keyed by (embedding slug, owner): switching models never reuses an index.
_vectorstore_cache: Dict[Tuple[str, str], FAISS] = {}
# Bumped whenever an owner's index changes; lets derived caches (the
# BM25 keyword index) know when to rebuild.
_versions: Dict[str, int] = {}


class VectorStoreError(Exception):
    """Raised for unrecoverable vector store operations."""


class IndexModelMismatch(VectorStoreError):
    """The user's index was built with a different embedding model."""


def embedding_slug(provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """Directory name for an embedding model, e.g. 'ollama__bge-m3'."""
    if provider is None or model is None:
        provider, model = embedding_choice()
    return f"{provider}__{re.sub(r'[^A-Za-z0-9._-]', '-', model)}"


def _check_owner(owner_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner_id or ""):
        raise VectorStoreError(f"Invalid owner id: {owner_id!r}")
    return owner_id


def _index_path(owner_id: str) -> Path:
    return settings.vector_store_dir / embedding_slug() / _check_owner(owner_id)


def _legacy_path(owner_id: str) -> Path:
    return settings.vector_store_dir / _check_owner(owner_id)


def _index_file(owner_id: str) -> Path:
    return _index_path(owner_id) / f"{_INDEX_NAME}.bin"


def _write_meta(owner_id: str, vs: FAISS) -> None:
    provider, model = embedding_choice()
    meta = {"provider": provider, "model": model, "dim": vs.index.d}
    (_index_path(owner_id) / _META_NAME).write_text(json.dumps(meta), encoding="utf-8")


def _reindex_hint(owner_id: str, built_with: str) -> IndexModelMismatch:
    return IndexModelMismatch(
        f"The index of user {owner_id} was built with {built_with}, but the current embedding "
        f"model is {embedding_slug()}. Run: python scripts/reindex.py --owner <username>"
    )


def _check_meta(owner_id: str) -> None:
    meta_file = _index_path(owner_id) / _META_NAME
    if not meta_file.exists():
        return
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if (meta.get("provider"), meta.get("model")) != embedding_choice():
        raise _reindex_hint(owner_id, f"{meta.get('provider')}/{meta.get('model')}")


def _other_model_indexes(owner_id: str) -> List[str]:
    """Embedding slugs (other than the current one) holding an index of
    this user; a pre-layout index counts as OpenAI's."""
    current = embedding_slug()
    found = []
    root = settings.vector_store_dir
    if root.exists():
        for model_dir in root.iterdir():
            index = model_dir / owner_id / f"{_INDEX_NAME}.bin"
            if model_dir.name != current and "__" in model_dir.name and index.exists():
                found.append(model_dir.name)
    if (_legacy_path(owner_id) / f"{_INDEX_NAME}.bin").exists():
        found.append(embedding_slug("openai", settings.openai_embedding_model) + " (legacy layout)")
    return found


def _migrate_legacy(owner_id: str) -> None:
    """Move a pre-layout index (built with OpenAI embeddings) into the
    current model's directory - only in openai mode, where it fits."""
    legacy = _legacy_path(owner_id)
    if embedding_choice()[0] != "openai" or not (legacy / f"{_INDEX_NAME}.bin").exists():
        return
    target = _index_path(owner_id)
    if (target / f"{_INDEX_NAME}.bin").exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy), str(target))
    logger.info("Moved the index of %s to %s", owner_id, target)


def _load_from_disk(owner_id: str) -> Optional[FAISS]:
    _migrate_legacy(owner_id)
    index_file = _index_file(owner_id)
    if not index_file.exists():
        others = _other_model_indexes(owner_id)
        if others:
            raise _reindex_hint(owner_id, ", ".join(others))
        return None
    _check_meta(owner_id)
    try:
        return FAISS.deserialize_from_bytes(
            index_file.read_bytes(),
            get_embeddings(),
            allow_dangerous_deserialization=True,  # we only load files this app wrote
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load FAISS index for %s from disk: %s", owner_id, exc)
        return None


def _save_to_disk(owner_id: str, vs: FAISS) -> None:
    # FAISS's own save_local/load_local open files from C++, which on
    # Windows fails for paths with non-ASCII characters (e.g. a project
    # under a Chinese folder name). Serializing to bytes and letting
    # Python do the file I/O works with any path.
    index_file = _index_file(owner_id)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = index_file.with_suffix(".tmp")
    tmp_file.write_bytes(vs.serialize_to_bytes())
    tmp_file.replace(index_file)  # atomic swap: a crash never leaves a half-written index
    _write_meta(owner_id, vs)


def _mark_changed(owner_id: str) -> None:
    _versions[owner_id] = _versions.get(owner_id, 0) + 1


def index_version(owner_id: str) -> int:
    return _versions.get(owner_id, 0)


def get_vectorstore(owner_id: str) -> Optional[FAISS]:
    """Return the owner's FAISS vector store, loading it from disk on
    first access. Returns None if the owner has not indexed anything."""
    key = (embedding_slug(), owner_id)
    with _lock:
        if key not in _vectorstore_cache:
            vs = _load_from_disk(owner_id)
            if vs is None:
                return None
            _vectorstore_cache[key] = vs
        return _vectorstore_cache[key]


def add_document_chunks(owner_id: str, chunks: List[LCDocument]) -> List[str]:
    """Embed and add a batch of chunks (typically all chunks belonging
    to one uploaded document) to the owner's vector store, persisting
    the updated index to disk. Returns the generated chunk ids."""
    if not chunks:
        return []

    with _lock:
        vs = get_vectorstore(owner_id)
        if vs is None:
            vs = FAISS.from_documents(chunks, get_embeddings())
            ids = list(vs.index_to_docstore_id.values())
            _vectorstore_cache[(embedding_slug(), owner_id)] = vs
        else:
            ids = vs.add_documents(chunks)
        _save_to_disk(owner_id, vs)
        _mark_changed(owner_id)
        return ids


def delete_document(owner_id: str, document_id: str) -> int:
    """Remove all chunks belonging to `document_id` from the owner's
    vector store. Returns the number of chunks removed."""
    with _lock:
        vs = get_vectorstore(owner_id)
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
        _save_to_disk(owner_id, vs)
        _mark_changed(owner_id)
        return len(ids_to_remove)


def delete_owner_index(owner_id: str) -> None:
    """Drop an owner's index for the current embedding model (memory and
    disk)."""
    with _lock:
        _vectorstore_cache.pop((embedding_slug(), owner_id), None)
        shutil.rmtree(_index_path(owner_id), ignore_errors=True)
        _mark_changed(owner_id)


def rebuild_index(owner_id: str, chunks: List[LCDocument]) -> int:
    """Replace the owner's index for the current embedding model with
    `chunks` (scripts/reindex.py). Indexes of other models are kept, so
    switching back needs no re-embedding. Returns the chunk count."""
    with _lock:
        _vectorstore_cache.pop((embedding_slug(), owner_id), None)
        shutil.rmtree(_index_path(owner_id), ignore_errors=True)
        if chunks:
            vs = FAISS.from_documents(chunks, get_embeddings())
            _save_to_disk(owner_id, vs)
            _vectorstore_cache[(embedding_slug(), owner_id)] = vs
        _mark_changed(owner_id)
        return len(chunks)


def get_all_chunks(owner_id: str, document_id: Optional[str] = None) -> List[LCDocument]:
    """Every stored chunk of an owner (optionally of one document)."""
    vs = get_vectorstore(owner_id)
    if vs is None:
        return []
    docs = list(vs.docstore._dict.values())  # noqa: SLF001
    if document_id:
        docs = [d for d in docs if d.metadata.get("document_id") == document_id]
    return docs


def similarity_search_with_scores(
    owner_id: str,
    query: str,
    k: int = 4,
    document_id: Optional[str] = None,
    filters: Optional[dict] = None,
) -> List[Tuple[LCDocument, float]]:
    """Semantic search in the owner's index. Returns (chunk, cosine
    similarity) pairs, most similar first. Cosine similarity is 1 for
    identical meaning and around 0 for unrelated text. `filters` limits
    the search to chunks whose metadata matches (see chunk_filters)."""
    vs = get_vectorstore(owner_id)
    if vs is None or vs.index.ntotal == 0:
        return []

    search_kwargs = {}
    if filters:
        # Same as the document filter below: scan the whole index so the
        # filter can't cut matching chunks off. Free for a flat index.
        search_kwargs = {"filter": make_predicate(filters, document_id), "fetch_k": vs.index.ntotal}
    elif document_id:
        search_kwargs = {"filter": {"document_id": document_id}, "fetch_k": vs.index.ntotal}

    results = vs.similarity_search_with_score(query, k=k, **search_kwargs)
    # FAISS's default index returns the *squared* L2 distance. OpenAI
    # embeddings are unit-length, and for unit vectors
    # squared_distance = 2 - 2 * cosine, so cosine = 1 - distance / 2.
    return [(doc, 1.0 - float(distance) / 2.0) for doc, distance in results]


def reset_cache() -> None:
    """Clear the in-memory cache, forcing a reload from disk on next
    access. Mainly useful for tests."""
    with _lock:
        _vectorstore_cache.clear()
        _versions.clear()
