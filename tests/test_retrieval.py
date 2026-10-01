"""
tests/test_retrieval.py
-------------------------
The retrieval pipeline (vector + BM25 + RRF + thresholds + rerank),
run offline with the hashing embeddings from conftest.
"""

from langchain_core.documents import Document as LCDocument

from services import retriever as retriever_module
from services.retriever import reciprocal_rank_fusion, retrieve
from services.vector_store import add_document_chunks, delete_document


def _chunk(text: str, document_id: str, index: int, page: int = 1) -> LCDocument:
    return LCDocument(
        page_content=text,
        metadata={
            "document_id": document_id,
            "document_name": f"{document_id}.pdf",
            "page": page,
            "page_end": page,
            "chunk_index": index,
        },
    )


def test_rrf_prefers_chunks_ranked_by_both_lists():
    a, b, c = (_chunk(t, "d", i) for i, t in enumerate("abc"))
    # b is 2nd in both lists; a and c are each 1st in only one.
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert fused[0] is b
    assert {d.page_content for d in fused} == {"a", "b", "c"}


def test_users_only_search_their_own_index(fake_embeddings):
    add_document_chunks("alice", [_chunk("alice secret recipe for apple pie", "a1", 0)])
    add_document_chunks("bob", [_chunk("bob notes about apple pie baking", "b1", 0)])

    alice_hits = retrieve("alice", "apple pie", use_rerank=False)
    bob_hits = retrieve("bob", "apple pie", use_rerank=False)
    assert {d.metadata["document_id"] for d in alice_hits} == {"a1"}
    assert {d.metadata["document_id"] for d in bob_hits} == {"b1"}
    assert retrieve("carol", "apple pie") == []


def test_document_filter_finds_chunks_buried_under_other_documents(fake_embeddings):
    # 60 chunks from another document match the query better than the
    # target document's only chunk. The old "fetch 4*k then filter"
    # approach returned nothing here.
    noise = [_chunk(f"refund policy refund policy note {i}", "noise", i) for i in range(60)]
    target = [_chunk("our refund rules are listed below", "target", 0)]
    add_document_chunks("owner", noise + target)

    hits = retrieve("owner", "refund policy", document_id="target", use_rerank=False)
    assert [d.metadata["document_id"] for d in hits] == ["target"]


def test_unrelated_chunks_are_dropped_by_similarity_threshold(fake_embeddings):
    add_document_chunks("owner", [_chunk("quarterly revenue grew twelve percent", "d", 0)])
    assert retrieve("owner", "zebra migration patterns", hybrid=False, use_rerank=False) == []


def test_hybrid_search_finds_exact_codes(fake_embeddings):
    chunks = [_chunk(f"general product description text number {i}", "d", i) for i in range(20)]
    chunks.append(_chunk("error XJ4471 means the pump overheated", "d", 20))
    add_document_chunks("owner", chunks)

    hits = retrieve("owner", "what does XJ4471 mean", k=2, hybrid=True, use_rerank=False)
    assert hits[0].metadata["chunk_index"] == 20


def test_bm25_segments_chinese(fake_embeddings):
    add_document_chunks(
        "owner",
        [
            _chunk("本公司的退款政策是购买后三十天内全额退款", "d", 0),
            _chunk("员工手册规定了上下班时间", "d", 1),
        ],
    )
    hits = retrieve("owner", "退款政策", k=1, hybrid=True, use_rerank=False)
    assert hits[0].metadata["chunk_index"] == 0


def test_rerank_reorders_and_drops_low_scores(fake_embeddings, monkeypatch):
    add_document_chunks(
        "owner",
        [_chunk("apple pie recipe one", "d", 0), _chunk("apple pie recipe two", "d", 1)],
    )

    def fake_rerank(question, docs):
        scores = {0: 2, 1: 9}  # chunk 1 is the relevant one, chunk 0 is below the bar
        return sorted(
            [(d, scores[d.metadata["chunk_index"]]) for d in docs], key=lambda p: p[1], reverse=True
        )

    monkeypatch.setattr(retriever_module, "rerank", fake_rerank)
    hits = retrieve("owner", "apple pie recipe", use_rerank=True)
    assert [d.metadata["chunk_index"] for d in hits] == [1]
    assert hits[0].metadata["rerank_score"] == 9


def test_rerank_failure_falls_back_to_retrieval_order(fake_embeddings, monkeypatch):
    add_document_chunks("owner", [_chunk("apple pie recipe", "d", 0)])
    monkeypatch.setattr(retriever_module, "rerank", lambda q, docs: None)
    assert len(retrieve("owner", "apple pie", use_rerank=True)) == 1


def test_index_persists_under_non_ascii_path(fake_embeddings, monkeypatch, tmp_path):
    # FAISS's C++ file I/O can't open non-ASCII paths on Windows; the
    # project itself often lives under a Chinese folder name.
    from services import vector_store

    monkeypatch.setattr(vector_store.settings, "vector_store_dir", tmp_path / "向量库")
    add_document_chunks("owner", [_chunk("persisted apple pie", "d", 0)])

    vector_store.reset_cache()  # force a reload from disk
    hits = retrieve("owner", "apple pie", use_rerank=False)
    assert [d.page_content for d in hits] == ["persisted apple pie"]


def test_deleted_document_disappears_from_both_searches(fake_embeddings):
    add_document_chunks("owner", [_chunk("kiwi smoothie XK12", "gone", 0), _chunk("banana bread", "stay", 0)])
    assert retrieve("owner", "kiwi smoothie XK12", use_rerank=False)

    assert delete_document("owner", "gone") == 1
    hits = retrieve("owner", "kiwi smoothie XK12", use_rerank=False)
    assert all(d.metadata["document_id"] != "gone" for d in hits)
