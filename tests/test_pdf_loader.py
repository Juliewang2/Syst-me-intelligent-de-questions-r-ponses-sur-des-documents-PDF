"""
tests/test_pdf_loader.py
--------------------------
Text extraction, including OCR of scanned (image-only) pages.
"""

import pytest
from PIL import Image, ImageDraw, ImageFont

from services.pdf_loader import PDFExtractionError, extract_pdf_text


def _scanned_pdf(path, lines):
    """Write a PDF whose single page is just a picture of `lines` - like
    a scan, it has no text layer for pypdf to read."""
    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=44)
    for i, line in enumerate(lines):
        draw.text((100, 150 + i * 90), line, font=font, fill="black")
    image.save(path, "PDF", resolution=150)
    return path


def test_scanned_page_is_read_with_ocr(tmp_path):
    pdf = _scanned_pdf(tmp_path / "scan.pdf", ["Invoice number 58213", "Total due: 420 EUR"])

    extracted = extract_pdf_text(pdf)

    assert extracted.ocr_page_count == 1
    assert "58213" in extracted.full_text
    assert "420" in extracted.full_text


def test_blank_scan_still_fails_cleanly(tmp_path):
    pdf = _scanned_pdf(tmp_path / "blank.pdf", [])
    with pytest.raises(PDFExtractionError):
        extract_pdf_text(pdf)
