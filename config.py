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
from typing import List, Literal, Optional

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

    # --- Deployment mode / model providers (services/llm_factory.py) ---
    # openai    : every model call and embedding goes to OpenAI (as before).
    # mistral   : every model call and embedding goes to Mistral.
    # souverain : everything runs locally through Ollama.
    # hybride   : chat models are external (OpenAI by default) for public
    #             context, local for restricted (DR) context; embeddings
    #             are always local.
    deployment_mode: Literal["openai", "mistral", "souverain", "hybride"] = Field(default="openai")
    # Per-role overrides of the mode's provider ("openai" | "mistral" |
    # "ollama") and model name. Empty = the mode's default.
    generate_provider: str = Field(default="")
    generate_model: str = Field(default="")
    rerank_provider: str = Field(default="")
    rerank_model: str = Field(default="")
    rewrite_provider: str = Field(default="")
    rewrite_model: str = Field(default="")
    verify_provider: str = Field(default="")
    verify_model: str = Field(default="")
    agent_provider: str = Field(default="")
    agent_model: str = Field(default="")
    mistral_api_key: str = Field(default="")
    mistral_chat_model: str = Field(default="mistral-small-latest")
    mistral_embedding_model: str = Field(default="mistral-embed")
    ollama_base_url: str = Field(default="http://localhost:11434")
    local_chat_model: str = Field(default="qwen2.5:7b")
    local_embedding_model: str = Field(default="bge-m3")
    # Local models are slower than hosted ones, so they get their own timeout.
    local_timeout_seconds: float = Field(default=120.0, gt=0)
    # Providers allowed to receive restricted (DR_SIMULE) text. A model call
    # whose context is restricted is rerouted to Ollama if its provider is
    # not listed; a restricted document is refused at import if the
    # embedding provider is not listed.
    dr_allowed_providers: List[str] = Field(default_factory=lambda: ["ollama"])

    # --- Evaluation ---
    # Model used by evaluation/run_eval.py to grade answers. Empty = use
    # OPENAI_CHAT_MODEL. Keeping the judge fixed means changing the model
    # under test doesn't silently change the grading standard too.
    judge_model: str = Field(default="")

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

    # --- Corrective retrieval (services/corrective_retriever.py) ---
    # When the best rerank score of a search is low, search again with
    # rewritten queries (expanded jargon, translations). Off = unchanged.
    corrective_retrieval_enabled: bool = Field(default=False)
    # Best score >= upper: results are good. Below lower: nothing relevant.
    # In between: keep what passed and add results of the rewrites.
    corrective_upper_threshold: int = Field(default=7, ge=0, le=10)
    # Empty = RERANK_MIN_SCORE.
    corrective_lower_threshold: Optional[int] = Field(default=None, ge=0, le=10)
    corrective_max_retries: int = Field(default=1, ge=0, le=3)

    # --- Diplomatic corpus ---
    # Read document metadata (sidecar .json / upload field) and copy it onto
    # chunks so retrieval can filter on it. Off = documents and chunks are
    # stored exactly as before. When on, a document without metadata is
    # treated as DR_SIMULE (restricted) - the safe default.
    corpus_metadata_enabled: bool = Field(default=False)
    public_corpus_dir: Path = Field(default=BASE_DIR / "data" / "corpus" / "public")
    un_votes_db_path: Path = Field(default=BASE_DIR / "data" / "un_votes.sqlite")

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
