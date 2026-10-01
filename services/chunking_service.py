"""
services/chunking_service.py
------------------------------
Splits extracted PDF text into overlapping chunks suitable for
embedding, using LangChain's `RecursiveCharacterTextSplitter`. Each
chunk is returned as a LangChain `Document` with rich metadata
(document id, filename, page number, chunk index) so the retriever
can later cite exact sources.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import List

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from services.pdf_loader import ExtractedPDF

settings = get_settings()

# Pages are joined with a single newline: a page break is usually not a
# paragraph break (text often continues mid-sentence onto the next page).
PAGE_JOINER = "\n"


def build_text_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    """Create a configured RecursiveCharacterTextSplitter.

    The recursive splitter tries a prioritized list of separators
    (paragraph, then line, then sentence, then word) so chunks break
    at natural language boundaries whenever possible, which keeps
    each chunk semantically coherent for embedding.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "。", "？", "！", "；", " ", ""],
        # Keep punctuation with the sentence it ends, not the next chunk's start.
        keep_separator="end",
        add_start_index=True,  # records each chunk's offset, used to map it back to pages
    )


def chunk_document(
    extracted: ExtractedPDF,
    document_id: str,
    filename: str,
) -> List[LCDocument]:
    """
    Convert an ExtractedPDF into a flat list of LangChain Documents,
    one per chunk, each carrying metadata used later for citations.

    All pages are joined into one continuous text before splitting, so
    a sentence or paragraph that runs over a page break stays in one
    chunk. We remember where each page starts in that text; from a
    chunk's start/end offsets we can then tell which page(s) it spans
    (`page` .. `page_end`).
    """
    page_texts: List[str] = []
    page_numbers: List[int] = []
    page_starts: List[int] = []
    offset = 0
    for page in extracted.pages:
        if not page.text or not page.text.strip():
            continue
        page_starts.append(offset)
        page_numbers.append(page.page_number)
        page_texts.append(page.text)
        offset += len(page.text) + len(PAGE_JOINER)

    if not page_texts:
        return []

    full_text = PAGE_JOINER.join(page_texts)
    splitter = build_text_splitter()
    pieces = splitter.create_documents([full_text])  # add_start_index=True

    def page_at(char_offset: int) -> int:
        return page_numbers[bisect_right(page_starts, char_offset) - 1]

    all_chunks: List[LCDocument] = []
    for piece in pieces:
        chunk_text = piece.page_content
        if not chunk_text.strip():
            continue
        start = piece.metadata["start_index"]
        end = start + len(chunk_text) - 1
        metadata = {
            "document_id": document_id,
            "document_name": filename,
            "page": page_at(start),
            "page_end": page_at(end),
            "chunk_index": len(all_chunks),
        }
        all_chunks.append(LCDocument(page_content=chunk_text, metadata=metadata))

    return all_chunks
