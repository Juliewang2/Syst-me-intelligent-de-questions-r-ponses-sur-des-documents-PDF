"""
tests/test_index_layout.py
----------------------------
One index directory per embedding model: metadata written with the
index, an error (pointing to scripts/reindex.py) when the model doesn't
match, the move of pre-layout indexes in openai mode, and rebuild_index.
"""

import json
import shutil

import pytest
from langchain_core.documents import Document as LCDocument

from services import vector_store
from services.retriever import retrieve
from services.vector_store import IndexModelMismatch, add_document_chunks, rebuild_index


def _chunk(text: str, index: int = 0) -> LCDocument:
    return LCDocument(
        page_content=text,
        metadata={"document_id": "d", "document_name": "d.pdf", "page": 1, "page_end": 1, "chunk_index": index},
    )


def _use_mode(monkeypatch, mode: str) -> None:
    monkeypatch.setattr(vector_store.settings, "deployment_mode", mode)
    vector_store.reset_cache()


def test_index_lives_under_its_model_with_metadata(fake_embeddings):
    add_document_chunks("owner", [_chunk("apple pie")])
    model_dir = vector_store.settings.vector_store_dir / "openai__text-embedding-3-small" / "owner"
    assert (model_dir / "faiss_index.bin").exists()
    meta = json.loads((model_dir / "index_meta.json").read_text())
    assert (meta["provider"], meta["model"]) == ("openai", "text-embedding-3-small")


def test_switching_embedding_model_requires_reindex(fake_embeddings, monkeypatch):
    add_document_chunks("owner", [_chunk("apple pie")])
    _use_mode(monkeypatch, "hybride")  # embeddings become ollama/bge-m3
    with pytest.raises(IndexModelMismatch, match="reindex.py"):
        retrieve("owner", "apple pie", use_rerank=False)


def test_metadata_naming_another_model_is_an_error(fake_embeddings):
    add_document_chunks("owner", [_chunk("apple pie")])
    meta_file = vector_store._index_path("owner") / "index_meta.json"  # noqa: SLF001
    meta_file.write_text(json.dumps({"provider": "openai", "model": "text-embedding-3-large", "dim": 3072}))
    vector_store.reset_cache()
    with pytest.raises(IndexModelMismatch, match="text-embedding-3-large"):
        retrieve("owner", "apple pie", use_rerank=False)


def test_pre_layout_index_is_moved_in_openai_mode(fake_embeddings):
    add_document_chunks("owner", [_chunk("legacy apple pie")])
    root = vector_store.settings.vector_store_dir
    # Recreate the old layout: vector_store_dir/<owner>/faiss_index.bin, no metadata.
    shutil.move(str(root / "openai__text-embedding-3-small" / "owner"), str(root / "owner"))
    (root / "owner" / "index_meta.json").unlink()
    vector_store.reset_cache()

    hits = retrieve("owner", "apple pie", use_rerank=False)
    assert [d.page_content for d in hits] == ["legacy apple pie"]
    assert not (root / "owner").exists()
    assert (root / "openai__text-embedding-3-small" / "owner" / "faiss_index.bin").exists()


def test_pre_layout_index_is_not_used_by_local_embeddings(fake_embeddings, monkeypatch):
    add_document_chunks("owner", [_chunk("legacy apple pie")])
    root = vector_store.settings.vector_store_dir
    shutil.move(str(root / "openai__text-embedding-3-small" / "owner"), str(root / "owner"))
    _use_mode(monkeypatch, "souverain")
    with pytest.raises(IndexModelMismatch, match="legacy layout"):
        retrieve("owner", "apple pie", use_rerank=False)
    assert (root / "owner" / "faiss_index.bin").exists()  # left in place


def test_rebuild_index_replaces_only_the_current_model(fake_embeddings, monkeypatch):
    add_document_chunks("owner", [_chunk("old openai text")])
    _use_mode(monkeypatch, "hybride")
    assert rebuild_index("owner", [_chunk("rebuilt local text")]) == 1

    assert [d.page_content for d in retrieve("owner", "rebuilt local text", use_rerank=False)] == [
        "rebuilt local text"
    ]
    _use_mode(monkeypatch, "openai")  # the OpenAI index is still there
    assert [d.page_content for d in retrieve("owner", "old openai text", use_rerank=False)] == ["old openai text"]
