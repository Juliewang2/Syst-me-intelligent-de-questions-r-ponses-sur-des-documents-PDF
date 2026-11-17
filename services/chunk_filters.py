"""
services/chunk_filters.py
---------------------------
Metadata filters for retrieval: restrict a search to public or internal
documents, some document types, one classification, or a date range.

One predicate (chunk metadata -> bool) is built here and shared by both
halves of hybrid search, so vector and BM25 results can never disagree
on what is in scope. Both apply it to the *whole* index before ranking -
the same approach as the existing per-document filter - so a filter
never empties the results just because the matching chunks ranked below
other documents' chunks.

Dates are stored on chunks as ISO strings (YYYY-MM-DD), so a range check
is a plain string comparison. A chunk lacking a filtered field (e.g. one
ingested before metadata was enabled) does not match that filter.
"""

from __future__ import annotations

import datetime
from typing import Callable, List, Optional

from pydantic import BaseModel, ConfigDict

Predicate = Callable[[dict], bool]


class ChunkFilters(BaseModel):
    # Unknown keys are an error, so a typo can't silently disable a filter.
    model_config = ConfigDict(extra="forbid")

    source: Optional[str] = None
    doc_type: Optional[List[str]] = None
    classification: Optional[str] = None
    date_from: Optional[datetime.date] = None
    date_to: Optional[datetime.date] = None

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())

    def matches(self, metadata: dict) -> bool:
        if self.source is not None and metadata.get("source") != self.source:
            return False
        if self.doc_type is not None and metadata.get("doc_type") not in self.doc_type:
            return False
        if self.classification is not None and metadata.get("classification") != self.classification:
            return False
        return self._date_in_range(metadata.get("date"))

    def _date_in_range(self, date: Optional[str]) -> bool:
        if self.date_from is None and self.date_to is None:
            return True
        if not date:
            return False
        if self.date_from is not None and date < self.date_from.isoformat():
            return False
        return self.date_to is None or date <= self.date_to.isoformat()


def make_predicate(filters: Optional[dict], document_id: Optional[str] = None) -> Optional[Predicate]:
    """A metadata predicate for the given filters (and optional document
    scope), or None when there is nothing to filter on. Invalid filters
    raise pydantic.ValidationError."""
    parsed = ChunkFilters.model_validate(filters or {})
    if parsed.is_empty() and not document_id:
        return None

    def predicate(metadata: dict) -> bool:
        if document_id and metadata.get("document_id") != document_id:
            return False
        return parsed.matches(metadata)

    return predicate
