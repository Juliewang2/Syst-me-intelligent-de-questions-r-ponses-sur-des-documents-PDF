"""
services/retriever.py
------------------------
The retrieval pipeline the RAG chain calls to fetch context for a
question. Given a question it:

  1. Vector search    - top candidates by meaning (FAISS), dropping any
                        below MIN_VECTOR_SIMILARITY (cosine).
  2. Keyword search   - top candidates by exact words (BM25), if hybrid
                        search is enabled.
  3. Fusion           - merges the two ranked lists with Reciprocal Rank
                        Fusion (RRF).
  4. Rerank           - the chat model scores each candidate 0-10.
  5. Threshold, top-k - candidates below RERANK_MIN_SCORE are dropped and
                        the best `k` chunks go into the prompt.

`retrieve_scored` runs steps 1-4 and returns every candidate with its
score (used by services/corrective_retriever.py); `retrieve` adds step 5.

Every step is switchable (settings, or per-call overrides used by the
evaluation script to compare configurations). Optional metadata
`filters` (source, doc_type, classification, date range - see
services/chunk_filters.py) apply to both searches before ranking. The pipeline is also
exposed as a LangChain `BaseRetriever` so it plugs into LCEL chains.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from config import get_settings
from services.keyword_search import keyword_search
from services.reranker import rerank
from services.vector_store import similarity_search_with_scores

settings = get_settings()

# Standard RRF constant: dampens the gap between rank 1 and rank 2 so a
# chunk ranked well by *both* searches beats one ranked first by only one.
_RRF_K = 60


def _chunk_key(doc: LCDocument) -> Tuple[Optional[str], Optional[int]]:
    return doc.metadata.get("document_id"), doc.metadata.get("chunk_index")


def reciprocal_rank_fusion(ranked_lists: List[List[LCDocument]]) -> List[LCDocument]:
    """Merge several best-first lists into one. Each chunk scores
    sum(1 / (60 + rank)) over the lists it appears in. RRF only uses
    ranks, so it can combine searches whose raw scores aren't comparable
    (cosine similarity vs. BM25)."""
    scores: Dict[tuple, float] = {}
    docs: Dict[tuple, LCDocument] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = _chunk_key(doc)
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def _with_scores(doc: LCDocument, **scores) -> LCDocument:
    """Copy a chunk with retrieval scores added to its metadata (the
    stored chunk objects are shared, so never mutate them)."""
    return LCDocument(page_content=doc.page_content, metadata={**doc.metadata, **scores})


def retrieve_scored(
    owner_id: str,
    query: str,
    document_id: Optional[str] = None,
    filters: Optional[dict] = None,
    hybrid: Optional[bool] = None,
    use_rerank: Optional[bool] = None,
    n_candidates: Optional[int] = None,
) -> List[LCDocument]:
    """Steps 1-4 without any cut: every fused candidate (at most
    n_candidates), best first. When reranked, each carries
    metadata["rerank_score"] (0-10); when reranking is off or failed the
    key is absent (metadata.get gives None) and the order is the fused
    one. Corrective retrieval reads these scores to judge how good the
    retrieval was."""
    hybrid = settings.hybrid_search_enabled if hybrid is None else hybrid
    use_rerank = settings.rerank_enabled if use_rerank is None else use_rerank
    n_candidates = n_candidates or settings.retrieval_candidates

    # 1. Semantic search, with a similarity floor.
    vector_hits = [
        _with_scores(doc, vector_similarity=round(similarity, 4))
        for doc, similarity in similarity_search_with_scores(
            owner_id, query, k=n_candidates, document_id=document_id, filters=filters
        )
        if similarity >= settings.min_vector_similarity
    ]
    ranked_lists = [vector_hits]

    # 2. Keyword search.
    if hybrid:
        ranked_lists.append(
            [
                doc
                for doc, _ in keyword_search(
                    owner_id, query, k=n_candidates, document_id=document_id, filters=filters
                )
            ]
        )

    # 3. Fusion.
    candidates = reciprocal_rank_fusion(ranked_lists)[:n_candidates]

    # 4. Rerank (falls back to the fused order if the model call fails).
    if use_rerank and candidates:
        scored = rerank(query, candidates)
        if scored is not None:
            return [_with_scores(doc, rerank_score=score) for doc, score in scored]
    # Unscored: no rerank_score key, exactly as before (.get() gives None).
    return candidates


def above_threshold(docs: List[LCDocument], min_score: Optional[int] = None) -> List[LCDocument]:
    """Drop chunks whose rerank score is below the bar; unscored chunks
    (rerank off or failed) are all kept, in their order."""
    bar = settings.rerank_min_score if min_score is None else min_score
    return [d for d in docs if d.metadata.get("rerank_score") is None or d.metadata["rerank_score"] >= bar]


def retrieve(
    owner_id: str,
    query: str,
    document_id: Optional[str] = None,
    k: Optional[int] = None,
    hybrid: Optional[bool] = None,
    use_rerank: Optional[bool] = None,
    filters: Optional[dict] = None,
) -> List[LCDocument]:
    """Run the full retrieval pipeline and return up to k chunks, best
    first. `hybrid` / `use_rerank` override the settings when given;
    `filters` restricts both searches to matching chunks."""
    k = k or settings.retriever_k
    candidates = retrieve_scored(
        owner_id, query, document_id=document_id, filters=filters, hybrid=hybrid,
        use_rerank=use_rerank, n_candidates=max(settings.retrieval_candidates, k),
    )
    # 5. Relevance threshold, then top-k.
    return above_threshold(candidates)[:k]


class PDFChatRetriever(BaseRetriever):
    """LangChain-compatible wrapper around `retrieve`, bound to one
    user's documents and optionally scoped to a single document."""

    owner_id: str
    k: int = Field(default=settings.retriever_k)
    document_id: Optional[str] = Field(default=None)
    hybrid: Optional[bool] = Field(default=None)
    use_rerank: Optional[bool] = Field(default=None)
    filters: Optional[dict] = Field(default=None)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[LCDocument]:  # noqa: ANN001
        return retrieve(
            self.owner_id,
            query,
            document_id=self.document_id,
            k=self.k,
            hybrid=self.hybrid,
            use_rerank=self.use_rerank,
            filters=self.filters,
        )


def get_retriever(
    owner_id: str,
    document_id: Optional[str] = None,
    k: Optional[int] = None,
    hybrid: Optional[bool] = None,
    use_rerank: Optional[bool] = None,
    filters: Optional[dict] = None,
) -> PDFChatRetriever:
    """Factory for a scoped retriever instance."""
    return PDFChatRetriever(
        owner_id=owner_id,
        k=k or settings.retriever_k,
        document_id=document_id,
        hybrid=hybrid,
        use_rerank=use_rerank,
        filters=filters,
    )
