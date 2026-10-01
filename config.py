"""
config.py
---------
Centralized application configuration using Pydantic v2 Settings.

All configuration is loaded from environment variables (and a local
`.env` file during development). This module exposes a single cached
`get_settings()` accessor so configuration is read once and shared
across the whole application.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SECRET_KEY = "change-this-secret-key-in-production"


class Settings(BaseSettings):
    """Application-wide settings, populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = Field(default="PDF Chat")
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)
    app_version: str = Field(default="1.0.0")

    # --- OpenAI ---
    openai_api_key: str = Field(default="")
    openai_chat_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    # Without a timeout a stalled connection can hang a request indefinitely.
    openai_timeout_seconds: float = Field(default=60.0, gt=0)
    openai_max_retries: int = Field(default=2, ge=0, le=10)

    # --- Database ---
    database_url: str = Field(default=f"sqlite:///{BASE_DIR}/data/pdf_chat.db")

    # --- Storage paths ---
    upload_dir: Path = Field(default=BASE_DIR / "data" / "uploads")
    vector_store_dir: Path = Field(default=BASE_DIR / "data" / "vector_store")

    # --- RAG / Chunking ---
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    retriever_k: int = Field(default=4, ge=1, le=20)

    # --- Retrieval quality ---
    # Hybrid search = vector (semantic) + BM25 (keyword), fused with RRF.
    hybrid_search_enabled: bool = Field(default=True)
    # How many candidates each retriever contributes before fusion/rerank.
    retrieval_candidates: int = Field(default=12, ge=1, le=100)
    # Vector hits below this cosine similarity are treated as irrelevant.
    min_vector_similarity: float = Field(default=0.2, ge=-1.0, le=1.0)
    # LLM-based reranking of the fused candidates (scores 0-10).
    rerank_enabled: bool = Field(default=True)
    rerank_min_score: int = Field(default=4, ge=0, le=10)
    # Reranking is optional (it falls back to the fused order), so it gets a
    # short timeout and no retries rather than holding up the answer.
    rerank_timeout_seconds: float = Field(default=20.0, gt=0)

    # --- OCR (for scanned PDFs without a text layer) ---
    ocr_enabled: bool = Field(default=True)
    # Pages whose extracted text is shorter than this are sent to OCR.
    ocr_min_chars: int = Field(default=20, ge=0)
    ocr_render_scale: float = Field(default=2.0, ge=0.5, le=5.0)

    # --- Uploads ---
    max_upload_size_mb: int = Field(default=25, ge=1, le=200)
    allowed_extensions: List[str] = Field(default_factory=lambda: [".pdf"])

    # --- CORS ---
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])

    # --- Security ---
    secret_key: str = Field(default=DEFAULT_SECRET_KEY)
    access_token_expire_minutes: int = Field(default=60 * 24 * 7, ge=5)
    # If set, new users must supply this code to register.
    registration_invite_code: str = Field(default="")

    # --- Conversation memory ---
    max_history_messages: int = Field(default=12, ge=2, le=100)

    @field_validator("openai_api_key")
    @classmethod
    def _warn_if_missing_key(cls, value: str) -> str:
        # We intentionally do not raise here so the app can still boot
        # (e.g. to serve the health check) without a key configured.
        return value

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create all directories required by the application at startup."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
