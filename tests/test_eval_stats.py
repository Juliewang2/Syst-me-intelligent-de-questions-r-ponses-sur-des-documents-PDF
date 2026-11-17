"""
tests/test_eval_stats.py
--------------------------
The evaluation's own arithmetic: the significance tests in
evaluation/stats.py and evidence-group matching in evaluation/evidence.py
(including legacy dataset conversion). Pure functions, no API calls.
"""

import pytest
from langchain_core.documents import Document as LCDocument

from evaluation.evidence import evidence_groups, first_hit_rank, groups_hit, normalize_item
from evaluation.stats import mcnemar_exact, paired_bootstrap


def _chunk(name: str, page: int, page_end: int = None, doc_id: str = None) -> LCDocument:
    metadata = {"document_name": name, "page": page, "page_end": page_end or page}
    if doc_id:
        metadata["doc_id"] = doc_id
    return LCDocument(page_content="text", metadata=metadata)


# --- paired bootstrap -------------------------------------------------------

def test_bootstrap_identical_samples_give_zero_difference():
    a = [0.2, 0.5, 0.9, 1.0]
    result = paired_bootstrap(a, list(a), n_resamples=500)
    assert result == {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_bootstrap_constant_shift_is_recovered():
    b = [0.1, 0.4, 0.3, 0.8, 0.6]
    a = [x + 0.25 for x in b]
    result = paired_bootstrap(a, b, n_resamples=500)
    assert result["mean_diff"] == pytest.approx(0.25)
    assert result["ci_low"] == pytest.approx(0.25)
    assert result["ci_high"] == pytest.approx(0.25)


def test_bootstrap_interval_contains_mean_and_is_reproducible():
    a = [1, 0, 1, 1, 0, 1, 1, 1]
    b = [0, 0, 1, 0, 0, 1, 0, 1]
    first = paired_bootstrap(a, b, n_resamples=2000, seed=7)
    assert first["ci_low"] <= first["mean_diff"] <= first["ci_high"]
    assert first["mean_diff"] == pytest.approx(3 / 8)
    assert paired_bootstrap(a, b, n_resamples=2000, seed=7) == first


def test_bootstrap_rejects_unpaired_samples():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


# --- McNemar -----------------------------------------------------------------

def test_mcnemar_counts_discordant_pairs():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    result = mcnemar_exact(a, b)
    assert (result["b01"], result["b10"]) == (1, 2)


def test_mcnemar_without_disagreement_has_p_one():
    assert mcnemar_exact([True, False], [True, False]) == {"b01": 0, "b10": 0, "p_value": 1.0}


def test_mcnemar_exact_p_value():
    # 6 disagreements, all in one direction: p = 2 * 0.5**6.
    a = [True] * 6 + [False] * 4
    b = [False] * 6 + [False] * 4
    assert mcnemar_exact(a, b)["p_value"] == pytest.approx(0.03125)


def test_mcnemar_is_symmetric():
    a = [True, True, True, False, True, False]
    b = [False, True, False, True, False, False]
    assert mcnemar_exact(a, b)["p_value"] == pytest.approx(mcnemar_exact(b, a)["p_value"])


# --- evidence groups ---------------------------------------------------------

def test_cross_file_question_needs_every_group():
    groups = [{"doc": "resolution.pdf", "pages": [2]}, {"doc": "opinion.pdf", "pages": [40]}]
    both = [_chunk("resolution.pdf", 2), _chunk("opinion.pdf", 39, 41)]
    one = [_chunk("resolution.pdf", 2), _chunk("opinion.pdf", 12)]

    assert groups_hit(both, groups) == [True, True]
    assert groups_hit(one, groups) == [True, False]
    assert first_hit_rank(one, groups) == 1


def test_group_without_pages_accepts_any_page_of_that_file():
    groups = [{"doc": "opinion.pdf", "pages": None}]
    assert groups_hit([_chunk("opinion.pdf", 87)], groups) == [True]
    assert groups_hit([_chunk("other.pdf", 87)], groups) == [False]


def test_doc_id_takes_precedence_over_file_name():
    chunk = _chunk("scan_0042.pdf", 3, doc_id="a-res-77-276-fr")
    assert groups_hit([chunk], [{"doc": "a-res-77-276-fr"}]) == [True]
    assert groups_hit([chunk], [{"doc": "scan_0042.pdf"}]) == [False]


def test_rank_counts_first_matching_chunk():
    groups = [{"doc": "a.pdf", "pages": [5]}]
    docs = [_chunk("a.pdf", 1), _chunk("b.pdf", 5), _chunk("a.pdf", 4, 6)]
    assert first_hit_rank(docs, groups) == 3
    assert first_hit_rank(docs[:2], groups) is None


# --- legacy format -----------------------------------------------------------

def test_legacy_pages_and_document_become_one_group():
    item = {"question": "q", "reference_answer": "r", "pages": [72], "document": "book.pdf", "evidence": "quote"}
    assert evidence_groups(item) == [{"doc": "book.pdf", "pages": [72], "evidence": "quote"}]


def test_legacy_pages_without_document_match_any_file():
    item = {"question": "q", "reference_answer": "r", "pages": [72]}
    groups = evidence_groups(item)
    assert groups == [{"doc": None, "pages": [72]}]
    assert groups_hit([_chunk("whatever.pdf", 72)], groups) == [True]


def test_new_format_is_kept_and_missing_pages_mean_whole_file():
    item = {"question": "q", "reference_answer": "r", "evidence_groups": [{"doc": "x.pdf"}]}
    assert normalize_item(item)["evidence_groups"] == [{"doc": "x.pdf", "pages": None}]


def test_unanswerable_question_has_no_groups():
    assert evidence_groups({"question": "q", "reference_answer": None, "pages": [3]}) == []
