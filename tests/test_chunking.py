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
