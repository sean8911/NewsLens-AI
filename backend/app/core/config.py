from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEWSLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    connect_timeout_seconds: float = Field(default=5, gt=0, le=120)
    read_timeout_seconds: float = Field(default=20, gt=0, le=120)
    retry_count: int = Field(default=2, ge=0, le=5)
    max_feed_bytes: int = Field(default=5 * 1024 * 1024, gt=0, le=50 * 1024 * 1024)
    user_agent: str = Field(default="NewsLensAI/0.1 (RSS collector)", min_length=1)
    feeds_path: Path | None = None
    database_url: SecretStr | None = None
    test_database_url: SecretStr | None = None
    database_connect_timeout_seconds: int = Field(default=5, ge=1, le=120)
    database_ssl: bool = False
    database_ssl_ca_file: Path | None = None
    database_statement_timeout_ms: int = Field(default=60000, ge=1, le=600000)
    database_lock_timeout_ms: int = Field(default=5000, ge=1, le=60000)
    database_idle_transaction_timeout_ms: int = Field(default=300000, ge=1000, le=1200000)
    api_max_body_bytes: int = Field(default=16384, ge=1024, le=1048576)
    api_body_timeout_seconds: float = Field(default=10, ge=0.1, le=120)
    api_requests_per_minute: int = Field(default=120, ge=1, le=10000)
    api_inference_requests_per_minute: int = Field(default=30, ge=1, le=1000)
    api_max_clients: int = Field(default=1024, ge=1, le=100000)
    api_max_inflight_inference: int = Field(default=2, ge=1, le=10)
    database_schema: str = Field(default="public", max_length=63, pattern=r"^[a-z_][a-z0-9_]*$")
    # No scheduler starts at import time; this flag only gates the explicit run command.
    scheduler_enabled: bool = True
    scheduler_interval_seconds: int = Field(default=600, ge=1, le=86400)
    scheduler_run_immediately: bool = True
    scheduler_misfire_grace_seconds: int = Field(default=60, ge=1, le=3600)
    content_web_enabled: bool = False
    content_policies_path: Path | None = None
    content_max_pages_per_run: int = Field(default=20, ge=1, le=200)
    content_max_bytes: int = Field(default=2 * 1024 * 1024, ge=1024, le=10 * 1024 * 1024)
    content_min_chars: int = Field(default=300, ge=100, le=5000)
    topic_model_path: Path = Path(".local/models/topics")
    entity_model_path: Path = Path(".local/models/entities")
    embedding_model_path: Path = Path(".local/models/embeddings")
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = Field(
        default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", pattern=r"^[0-9a-f]{40}$"
    )
    embeddings_enabled: bool = True
    multilingual_enabled: bool = True
    multilingual_embedding_model_id: str = "intfloat/multilingual-e5-small"
    multilingual_embedding_model_revision: str = Field(
        default="614241f622f53c4eeff9890bdc4f31cfecc418b3", pattern=r"^[0-9a-f]{40}$"
    )
    multilingual_reranking_model_id: str = "BAAI/bge-reranker-v2-m3"
    multilingual_reranking_model_revision: str = Field(
        default="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", pattern=r"^[0-9a-f]{40}$"
    )
    # BGE raw logits are not calibrated like the legacy MS MARCO reranker.
    multilingual_chat_min_reranking_score: float = Field(
        default=-9, ge=-100, le=100, allow_inf_nan=False
    )
    translation_model_path: Path = Path(".local/models/translation")
    translation_model_id: str = "facebook/m2m100_418M"
    translation_model_revision: str = Field(
        default="55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636", pattern=r"^[0-9a-f]{40}$"
    )
    events_enabled: bool = True
    event_summaries_enabled: bool = True
    event_similarity_threshold: float = Field(default=0.78, gt=0, lt=1, allow_inf_nan=False)
    event_window_hours: float = Field(default=72, gt=0, le=720, allow_inf_nan=False)
    event_max_articles: int = Field(default=2000, ge=2, le=5000)
    trending_max_articles: int = Field(default=10_000, ge=100, le=50_000)
    reranking_model_path: Path = Path(".local/models/reranking")
    reranking_model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranking_model_revision: str = Field(
        default="233902d25c440f23af6f7d6e94d2946bac0bee0a", pattern=r"^[0-9a-f]{40}$"
    )
    reranking_candidate_limit: int = Field(default=30, ge=20, le=30)
    chat_model_path: Path = Path(".local/models/chat")
    chat_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    chat_model_revision: str = Field(
        default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306", pattern=r"^[0-9a-f]{40}$"
    )
    chat_min_reranking_score: float = Field(default=0, ge=-100, le=100, allow_inf_nan=False)
    chat_max_input_tokens: int = Field(default=4096, ge=1024, le=8192)
    chat_conversation_max_input_tokens: int = Field(default=1024, ge=1024, le=8192)
    chat_conversation_generation_seconds: int = Field(default=90, ge=1, le=120)
    chat_generation_seconds: int = Field(default=45, ge=1, le=120)
    research_input_tokens: int = Field(default=1024, ge=1024, le=4096)
    research_generation_seconds: int = Field(default=90, ge=1, le=120)
    research_seconds: int = Field(default=600, ge=10, le=1200)
    recency_decay: Literal["exponential", "gaussian", "hyperbolic"] = "exponential"
    recency_half_life_days: float = Field(default=7, gt=0, le=3650, allow_inf_nan=False)
    recency_strength: float = Field(default=0.65, ge=0, le=1, allow_inf_nan=False)
    recency_reranking_weight: float = Field(default=0.85, ge=0, le=1, allow_inf_nan=False)
    hybrid_candidate_limit: int = Field(default=50, ge=1, le=50)
    hybrid_rrf_k: int = Field(default=60, ge=1, le=1000)
    hybrid_semantic_weight: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)
    hybrid_lexical_weight: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)
    qdrant_path: Path = Path(".local/qdrant")
    qdrant_search_wait_seconds: float = Field(default=10, ge=0, le=120)
    qdrant_update_wait_seconds: float = Field(default=120, ge=0, le=600)
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_prefix: str = Field(
        default="newslens_articles", max_length=80, pattern=r"^[a-z][a-z0-9_]*$"
    )
