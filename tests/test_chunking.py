"""
tests/test_chunking.py
------------------------
Unit tests for the chunking service that don't require any network
or OpenAI calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.chunking_service import build_text_splitter, chunk_document  # noqa: E402
from services.pdf_loader import ExtractedPage, ExtractedPDF  # noqa: E402


def test_text_splitter_respects_chunk_size():
    splitter = build_text_splitter(chunk_size=100, chunk_overlap=20)
    long_text = "This is a sentence. " * 50
    chunks = splitter.split_text(long_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100 + 20  # allow small overlap slack


def test_chunk_document_produces_metadata():
    extracted = ExtractedPDF(
        pages=[
            ExtractedPage(page_number=1, text="Alpha beta gamma. " * 30),
            ExtractedPage(page_number=2, text="Delta epsilon zeta. " * 30),
        ],
        page_count=2,
    )

    chunks = chunk_document(extracted, document_id="doc-123", filename="sample.pdf")

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["document_id"] == "doc-123"
        assert chunk.metadata["document_name"] == "sample.pdf"
        assert chunk.metadata["page"] in (1, 2)
        assert chunk.metadata["page_end"] in (1, 2)
        assert isinstance(chunk.metadata["chunk_index"], int)


def test_chunk_document_skips_empty_pages():
    extracted = ExtractedPDF(
        pages=[
            ExtractedPage(page_number=1, text=""),
            ExtractedPage(page_number=2, text="Some real content here."),
        ],
        page_count=2,
    )

    chunks = chunk_document(extracted, document_id="doc-456", filename="sample2.pdf")
    assert all(c.metadata["page"] == 2 for c in chunks)


def test_sentence_across_page_break_stays_in_one_chunk():
    extracted = ExtractedPDF(
        pages=[
            ExtractedPage(page_number=1, text="Intro text. The refund window for all orders is"),
            ExtractedPage(page_number=2, text="thirty days from delivery. Other terms follow."),
        ],
        page_count=2,
    )

    chunks = chunk_document(extracted, document_id="doc-789", filename="terms.pdf")

    spanning = [c for c in chunks if "orders is\nthirty days" in c.page_content]
    assert len(spanning) == 1
    assert spanning[0].metadata["page"] == 1
    assert spanning[0].metadata["page_end"] == 2


def test_page_ranges_follow_chunk_offsets():
    extracted = ExtractedPDF(
        pages=[ExtractedPage(page_number=n, text=f"Page {n} sentence. " * 40) for n in (1, 2, 3)],
        page_count=3,
    )

    chunks = chunk_document(extracted, document_id="doc-1", filename="p.pdf")

    for chunk in chunks:
        first, last = chunk.metadata["page"], chunk.metadata["page_end"]
        assert 1 <= first <= last <= 3
        # The chunk's text really comes from those pages.
        assert f"Page {first} " in chunk.page_content
        assert f"Page {last} " in chunk.page_content
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].metadata["page_end"] == 3
