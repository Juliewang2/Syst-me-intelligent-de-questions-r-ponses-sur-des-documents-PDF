"""
services/keyword_search.py
-----------------------------
BM25 keyword search over a user's chunks - the "exact words" half of
hybrid retrieval.

Vector search matches *meaning*, which is great for paraphrases but
weak on exact tokens: product codes, names, IDs, rare terms. BM25 is
the classic search-engine ranking function: a chunk scores higher the
more often it contains the query's words, weighted so that rare words
count more than common ones and long chunks aren't favoured just for
being long.

Text is tokenized with jieba, which segments Chinese into words (it has
no spaces) and leaves English words intact. The BM25 index is built
lazily per user from the chunks already stored in their FAISS docstore,
and rebuilt only when that index changes.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from typing import Dict, List, Optional, Tuple

import jieba
from langchain_core.documents import Document as LCDocument
from rank_bm25 import BM25Okapi

from services.vector_store import get_all_chunks, index_version

jieba.setLogLevel(logging.WARNING)

# A token is kept if it has at least one letter, digit, or CJK character
# (drops whitespace and punctuation tokens produced by jieba).
_WORD_CHAR = re.compile(r"\w")

_lock = threading.Lock()
# owner_id -> (index version it was built from, chunks, bm25)
_cache: Dict[str, Tuple[int, List[LCDocument], "LuceneBM25"]] = {}


class LuceneBM25(BM25Okapi):
    """BM25 with the IDF formula used by Lucene / Elasticsearch.

    Classic Okapi IDF, log((N - n + 0.5) / (n + 0.5)), is zero or
    negative for any word found in half or more of the chunks - so in a
    short document with only a few chunks, keyword search would find
    nothing. Lucene's log(1 + (N - n + 0.5) / (n + 0.5)) keeps the same
    ranking behaviour but is always positive."""

    def _calc_idf(self, nd):
        self.idf = {
            word: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for word, freq in nd.items()
        }


def tokenize(text: str) -> List[str]:
    return [t for t in jieba.lcut_for_search(text.lower()) if _WORD_CHAR.search(t)]


def _get_index(owner_id: str) -> Optional[Tuple[List[LCDocument], BM25Okapi]]:
    version = index_version(owner_id)
    with _lock:
        cached = _cache.get(owner_id)
        if cached is not None and cached[0] == version:
            return cached[1], cached[2]

        chunks = get_all_chunks(owner_id)
        if not chunks:
            _cache.pop(owner_id, None)
            return None
        bm25 = LuceneBM25([tokenize(c.page_content) for c in chunks])
        _cache[owner_id] = (version, chunks, bm25)
        return chunks, bm25


def keyword_search(
    owner_id: str,
    query: str,
    k: int = 4,
    document_id: Optional[str] = None,
) -> List[Tuple[LCDocument, float]]:
    """Return up to k (chunk, BM25 score) pairs, best first. Chunks that
    share no words with the query (score 0) are never returned."""
    index = _get_index(owner_id)
    query_tokens = tokenize(query)
    if index is None or not query_tokens:
        return []
    chunks, bm25 = index

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    results: List[Tuple[LCDocument, float]] = []
    for i in ranked:
        if scores[i] <= 0 or len(results) >= k:
            break
        if document_id and chunks[i].metadata.get("document_id") != document_id:
            continue
        results.append((chunks[i], float(scores[i])))
    return results


def reset_cache() -> None:
    with _lock:
        _cache.clear()
