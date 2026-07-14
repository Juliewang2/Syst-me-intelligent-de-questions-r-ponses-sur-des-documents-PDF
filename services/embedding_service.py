"""
services/embedding_service.py
-------------------------------
Thin wrapper around LangChain's `OpenAIEmbeddings`, exposed as a
cached singleton so the embedding client (and its connection pool)
is reused across requests instead of being recreated on every call.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config import get_settings

settings = get_settings()


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings client configured from settings."""
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key or None,
    )
