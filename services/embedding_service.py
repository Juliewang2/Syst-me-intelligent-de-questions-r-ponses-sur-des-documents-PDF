"""
services/embedding_service.py
-------------------------------
The embedding model used to index and search chunks. Which model that
is depends on the deployment mode (OpenAI, Mistral, or local bge-m3 via
Ollama in hybrid and sovereign modes) and is decided by
services/llm_factory.py; this module keeps the import path the vector
store has always used.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from services import llm_factory


def get_embeddings() -> Embeddings:
    """The current deployment mode's (cached) embedding client."""
    return llm_factory.get_embeddings()
