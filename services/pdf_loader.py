"""
services/pdf_loader.py
-----------------------
Handles extraction of text (and basic metadata) from uploaded PDF
files. Uses LangChain Community's `PyPDFLoader` (backed by `pypdf`)
as the primary extraction path, with a `PyPDF2`-based fallback so the
application keeps working even if one of the two backends struggles
with a malformed PDF.

Scanned PDFs (pages that are just images, with no text layer) are
handled with OCR: any page whose extracted text is shorter than
`OCR_MIN_CHARS` is rendered to an image with `pypdfium2` and read with
RapidOCR (an ONNX port of PaddleOCR that handles Chinese and English).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LCDocument
from pypdf import PdfReader as PypdfReader
from PyPDF2 import PdfReader as PyPDF2Reader

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be parsed by any available backend."""


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    ocr: bool = False  # True if the text came from OCR rather than the text layer


@dataclass
class ExtractedPDF:
    pages: List[ExtractedPage]
    page_count: int

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def is_empty(self) -> bool:
        return self.page_count == 0 or not self.full_text.strip()

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.ocr)


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


@lru_cache
def _get_ocr_engine():
    """Load the OCR models once per process (takes a second or two)."""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _ocr_page(pdf, page_index: int) -> str:
    """Render one page to an image and run OCR on it."""
    bitmap = pdf[page_index].render(scale=settings.ocr_render_scale)
    image = bitmap.to_numpy()
    result, _elapsed = _get_ocr_engine()(image)
    if not result:
        return ""
    # Each item is [bounding_box, text, confidence], ordered top to bottom.
    return "\n".join(item[1] for item in result)


def _ocr_sparse_pages(file_path: Path, extracted: ExtractedPDF) -> None:
    """OCR every page whose text layer is (nearly) empty, in place."""
    sparse = [p for p in extracted.pages if len(p.text.strip()) < settings.ocr_min_chars]
    if not sparse:
        return

    import pypdfium2

    logger.info("Running OCR on %d page(s) of %s", len(sparse), file_path.name)
    pdf = pypdfium2.PdfDocument(str(file_path))
    try:
        for page in sparse:
            try:
                ocr_text = _ocr_page(pdf, page.page_number - 1)
            except Exception as exc:  # noqa: BLE001 - one bad page shouldn't fail the file
                logger.warning("OCR failed on page %d of %s: %s", page.page_number, file_path.name, exc)
                continue
            if len(ocr_text.strip()) > len(page.text.strip()):
                page.text = ocr_text
                page.ocr = True
    finally:
        pdf.close()


def extract_pdf_text(file_path: Path) -> ExtractedPDF:
    """
    Extract text from a PDF file, trying multiple backends in order of
    preference, then OCR any pages that have no usable text layer.
    Raises `PDFExtractionError` if all backends fail or the PDF yields
    no text even after OCR.
    """
    errors: list[str] = []
    extracted: ExtractedPDF | None = None

    for extractor in (_extract_with_langchain, _extract_with_pypdf, _extract_with_pypdf2):
        try:
            result = extractor(file_path)
        except Exception as exc:  # noqa: BLE001 - we want to try all backends
            logger.warning("PDF extraction backend %s failed: %s", extractor.__name__, exc)
            errors.append(f"{extractor.__name__} raised: {exc}")
            continue
        if extracted is None:
            extracted = result  # keep the page structure even if it has no text
        if not result.is_empty():
            extracted = result
            break
        errors.append(f"{extractor.__name__} produced no extractable text")

    if extracted is None:
        raise PDFExtractionError(
            "Unable to read PDF. This file may be corrupted or password-protected. "
            f"Details: {'; '.join(errors)}"
        )

    if settings.ocr_enabled:
        _ocr_sparse_pages(file_path, extracted)

    if extracted.is_empty():
        raise PDFExtractionError(
            "Unable to extract text from PDF, even with OCR. The pages may be blank "
            "or the scan too low-quality to read. "
            f"Details: {'; '.join(errors)}"
        )
    return extracted


def validate_pdf_header(file_path: Path) -> bool:
    """Quick sanity check that a file is actually a PDF (magic bytes)."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except OSError:
        return False
