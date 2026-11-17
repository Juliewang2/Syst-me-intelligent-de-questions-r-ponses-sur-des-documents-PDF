"""
evaluation/evidence.py
------------------------
Where a question's answer lives, and whether retrieval found it.

A question's expected evidence is a list of *evidence groups*, one per
source the answer needs:

    "evidence_groups": [
        {"doc": "a-res-77-276-fr", "pages": [2]},
        {"doc": "icj-ao-187-fr"}                  # no pages: any page counts
    ]

A multi-source question (a resolution plus the advisory opinion it
requested) is only fully answerable if *every* group is retrieved, which
a single "hit" can't express - hence per-group matching here, from which
run_eval.py derives hit, evidence_recall and all_evidence_hit.

`doc` is matched against a chunk's metadata["doc_id"] when the chunk has
one, otherwise against its document_name (the PDF file name). Older
datasets use `pages` (+ optional `document`) at the top level; they are
converted to a single group so both formats score identically.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from langchain_core.documents import Document as LCDocument


class EvidenceGroup(TypedDict, total=False):
    # None = any document (legacy items that never named their PDF).
    doc: Optional[str]
    # None = the whole document counts.
    pages: Optional[List[int]]
    # Optional quote that check_dataset.py verifies on those pages.
    evidence: str


def _legacy_group(item: dict) -> EvidenceGroup:
    group: EvidenceGroup = {"doc": item.get("document"), "pages": list(item["pages"])}
    if item.get("evidence"):
        group["evidence"] = item["evidence"]
    return group


def evidence_groups(item: dict) -> List[EvidenceGroup]:
    """The question's evidence groups, converting the legacy format.
    Unanswerable questions have none."""
    if item.get("reference_answer") is None:
        return []
    if "evidence_groups" in item:
        return [
            {**g, "doc": g.get("doc"), "pages": g.get("pages")} for g in item["evidence_groups"]
        ]
    if item.get("pages"):
        return [_legacy_group(item)]
    return []


def normalize_item(item: dict) -> dict:
    """A copy of a dataset item with `evidence_groups` filled in."""
    return {**item, "evidence_groups": evidence_groups(item)}


def chunk_doc_key(doc: LCDocument) -> Optional[str]:
    return doc.metadata.get("doc_id") or doc.metadata.get("document_name")


def chunk_matches_group(doc: LCDocument, group: EvidenceGroup) -> bool:
    if group.get("doc") is not None and chunk_doc_key(doc) != group["doc"]:
        return False
    pages = group.get("pages")
    if pages is None:
        return True
    first = doc.metadata.get("page")
    last = doc.metadata.get("page_end", first)
    if first is None:
        return False
    return any(first <= p <= last for p in pages)


def groups_hit(docs: List[LCDocument], groups: List[EvidenceGroup]) -> List[bool]:
    """For each group: did any retrieved chunk fall inside it?"""
    return [any(chunk_matches_group(d, g) for d in docs) for g in groups]


def first_hit_rank(docs: List[LCDocument], groups: List[EvidenceGroup]) -> Optional[int]:
    """1-based rank of the first chunk matching any group, or None."""
    for rank, doc in enumerate(docs, start=1):
        if any(chunk_matches_group(doc, g) for g in groups):
            return rank
    return None
