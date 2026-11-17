"""
tests/test_filters.py
-----------------------
Metadata filters on retrieval (source, doc_type, classification, date
range), checked separately on the vector search, the BM25 search and the
full pipeline, offline with the hashing embeddings from conftest.
"""

import pytest
from langchain_core.documents import Document as LCDocument
from pydantic import ValidationError

from services.keyword_search import keyword_search
from services.retriever import retrieve
from services.vector_store import add_document_chunks, similarity_search_with_scores


def _chunk(text: str, document_id: str, index: int, **meta) -> LCDocument:
    metadata = {
        "document_id": document_id,
        "document_name": f"{document_id}.pdf",
        "page": 1,
        "page_end": 1,
        "chunk_index": index,
        **meta,
    }
    return LCDocument(page_content=text, metadata=metadata)


@pytest.fixture
def corpus(fake_embeddings):
    add_document_chunks(
        "owner",
        [
            _chunk("climate resolution vote text", "res", 0, source="public", doc_type="resolution",
                   classification="PUBLIC", date="2023-03-29"),
            _chunk("climate advisory opinion text", "avis", 0, source="public", doc_type="avis_cij",
                   classification="PUBLIC", date="2025-07-23"),
            _chunk("climate instructions vote text", "td", 0, source="interne", doc_type="td",
                   classification="DR_SIMULE", date="2026-05-18"),
            _chunk("climate legacy text without metadata", "old", 0),
        ],
    )


def _ids(pairs) -> set:
    return {doc.metadata["document_id"] for doc, _ in pairs}


def _search_both(filters: dict) -> tuple:
    vector = _ids(similarity_search_with_scores("owner", "climate text", k=10, filters=filters))
    bm25 = _ids(keyword_search("owner", "climate text", k=10, filters=filters))
    return vector, bm25


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"source": "interne"}, {"td"}),
        ({"source": "public"}, {"res", "avis"}),
        ({"doc_type": ["resolution", "avis_cij"]}, {"res", "avis"}),
        ({"classification": "PUBLIC"}, {"res", "avis"}),
        ({"classification": "DR_SIMULE"}, {"td"}),
        ({"date_from": "2025-01-01"}, {"avis", "td"}),
        ({"date_to": "2025-07-23"}, {"res", "avis"}),  # bounds are inclusive
        ({"date_from": "2024-01-01", "date_to": "2025-12-31"}, {"avis"}),
        ({"source": "public", "date_from": "2024-01-01"}, {"avis"}),
    ],
)
def test_filters_apply_to_vector_and_bm25(corpus, filters, expected):
    vector, bm25 = _search_both(filters)
    assert vector == expected
    assert bm25 == expected


def test_no_filter_keeps_every_document(corpus):
    vector, bm25 = _search_both({})
    assert vector == bm25 == {"res", "avis", "td", "old"}


def test_full_pipeline_honours_filters(corpus):
    hits = retrieve("owner", "climate vote", k=4, use_rerank=False, filters={"classification": "PUBLIC"})
    assert hits
    assert {d.metadata["classification"] for d in hits} == {"PUBLIC"}


def test_filter_combines_with_document_scope(corpus):
    hits = retrieve("owner", "climate", use_rerank=False, document_id="res", filters={"source": "public"})
    assert {d.metadata["document_id"] for d in hits} == {"res"}
    assert retrieve("owner", "climate", use_rerank=False, document_id="res", filters={"source": "interne"}) == []


def test_filter_finds_chunks_buried_under_other_documents(fake_embeddings):
    # 60 public chunks match the query better than the one internal chunk;
    # filtering must still return it from both searches.
    noise = [_chunk(f"refund policy refund policy note {i}", "noise", i, source="public") for i in range(60)]
    target = [_chunk("our refund rules are listed below", "target", 0, source="interne")]
    add_document_chunks("owner", noise + target)

    vector = _ids(similarity_search_with_scores("owner", "refund policy", k=4, filters={"source": "interne"}))
    bm25 = _ids(keyword_search("owner", "refund policy", k=4, filters={"source": "interne"}))
    assert vector == bm25 == {"target"}


def test_unknown_filter_key_is_an_error(corpus):
    with pytest.raises(ValidationError):
        retrieve("owner", "climate", use_rerank=False, filters={"classificaton": "PUBLIC"})
