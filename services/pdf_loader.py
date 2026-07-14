"""
services/pdf_loader.py
-----------------------
Handles extraction of text (and basic metadata) from uploaded PDF
files. Uses LangChain Community's `PyPDFLoader` (backed by `pypdf`)
as the primary extraction path, with a `PyPDF2`-based fallback so the
application keeps working even if one of the two backends struggles
with a malformed PDF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LCDocument
from pypdf import PdfReader as PypdfReader
from PyPDF2 import PdfReader as PyPDF2Reader

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be parsed by any available backend."""


@dataclass
class ExtractedPage:
    page_number: int
    text: str


@dataclass
class ExtractedPDF:
    pages: List[ExtractedPage]
    page_count: int

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def is_empty(self) -> bool:
        return self.page_count == 0 or not self.full_text.strip()


def _extract_with_langchain(file_path: Path) -> ExtractedPDF:
    """Primary extraction path using LangChain's PyPDFLoader."""
    loader = PyPDFLoader(str(file_path))
    lc_docs: List[LCDocument] = loader.load()

    pages = [
        ExtractedPage(page_number=i + 1, text=doc.page_content or "")
        for i, doc in enumerate(lc_docs)
    ]
    return ExtractedPDF(pages=pages, page_count=len(pages))


def _extract_with_pypdf(file_path: Path) -> ExtractedPDF:
    """Secondary extraction path using pypdf directly."""
    reader = PypdfReader(str(file_path))
    pages = [
        ExtractedPage(page_number=i + 1, text=page.extract_text() or "")
        for i, page in enumerate(reader.pages)
    ]
    return ExtractedPDF(pages=pages, page_count=len(pages))


def _extract_with_pypdf2(file_path: Path) -> ExtractedPDF:
    """Tertiary fallback extraction path using PyPDF2."""
    reader = PyPDF2Reader(str(file_path))
    pages = [
        ExtractedPage(page_number=i + 1, text=page.extract_text() or "")
        for i, page in enumerate(reader.pages)
    ]
    return ExtractedPDF(pages=pages, page_count=len(pages))


def extract_pdf_text(file_path: Path) -> ExtractedPDF:
    """
    Extract text from a PDF file, trying multiple backends in order of
    preference. Raises `PDFExtractionError` if all backends fail or
    the PDF yields no extractable text (e.g. a pure image scan without
    OCR).
    """
    errors: list[str] = []

    for extractor in (_extract_with_langchain, _extract_with_pypdf, _extract_with_pypdf2):
        try:
            result = extractor(file_path)
            if not result.is_empty():
                return result
            errors.append(f"{extractor.__name__} produced no extractable text")
        except Exception as exc:  # noqa: BLE001 - we want to try all backends
            logger.warning("PDF extraction backend %s failed: %s", extractor.__name__, exc)
            errors.append(f"{extractor.__name__} raised: {exc}")

    raise PDFExtractionError(
        "Unable to extract text from PDF. This file may be a scanned image "
        "without a text layer, corrupted, or password-protected. "
        f"Details: {'; '.join(errors)}"
    )


def validate_pdf_header(file_path: Path) -> bool:
    """Quick sanity check that a file is actually a PDF (magic bytes)."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except OSError:
        return False
