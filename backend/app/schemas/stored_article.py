from datetime import datetime

from pydantic import BaseModel, ConfigDict

from backend.app.schemas.article import Article


class StoredTopic(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    topic: str
    confidence: float


class StoredEntity(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    canonical_name: str
    entity_type: str


class StoredKeyword(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    keyword: str
    canonical_name: str
    score: float
    rank: int


class StoredArticle(Article):
    model_config = ConfigDict(from_attributes=True)

    id: int
    content: str | None
    content_available: bool
    content_extraction_method: str
    content_extraction_status: str
    language: str | None
    language_confidence: float | None
    language_status: str
    language_detector: str | None
    language_text_source: str | None
    language_input_hash: str | None
    article_topics: list[StoredTopic]
    topic_status: str
    topic_model: str | None
    topic_input_hash: str | None
    topic_text_source: str | None
    topic_language: str | None
    topic_scores: dict[str, float]
    entities: list[StoredEntity]
    entity_status: str
    entity_model: str | None
    entity_input_hash: str | None
    entity_text_source: str | None
    entity_language: str | None
    entity_text_truncated: bool
    keywords: list[StoredKeyword]
    keyword_status: str
    keyword_model: str | None
    keyword_input_hash: str | None
    keyword_text_source: str | None
    keyword_language: str | None
    keyword_text_truncated: bool
    summary: str | None
    summary_one_sentence: str | None
    summary_bullets: list[str]
    summary_sentence_spans: list[dict[str, int]]
    summary_created_at: datetime | None
    summary_status: str
    summary_model: str | None
    summary_input_hash: str | None
    summary_text_source: str | None
    summary_language: str | None
    summary_source_chars: int
    summary_text_truncated: bool
    created_at: datetime
    updated_at: datetime
    cleaning_version: str | None
    cleaning_issues: list[str]
    processing_status: str
