from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Identity,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.base import Base
from backend.app.models.article_keyword import ArticleKeywordRecord
from backend.app.models.article_topic import ArticleTopicRecord
from backend.app.models.entity import EntityRecord, article_entities


class ArticleRecord(Base):
    __tablename__ = "articles"
    __table_args__ = (
        CheckConstraint("length(trim(title)) > 0", name="title_nonblank"),
        CheckConstraint(
            "(summary IS NULL AND summary_one_sentence IS NULL AND summary_created_at IS NULL "
            "AND summary_status <> 'summarized') OR "
            "(summary IS NOT NULL AND summary_one_sentence IS NOT NULL "
            "AND summary_created_at IS NOT NULL "
            "AND summary_status = 'summarized' AND length(summary) > 0 "
            "AND length(summary_one_sentence) > 0 AND length(summary) < summary_source_chars "
            "AND length(summary_one_sentence) <= length(summary))",
            name="summary_consistent",
        ),
        Index("ix_articles_source", "source"),
        Index("ix_articles_category", "category"),
        Index("ix_articles_published_at", "published_at"),
        Index("ix_articles_retrieved_at_id", "retrieved_at", "id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    content_available: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    content_extraction_method: Mapped[str] = mapped_column(String(32), server_default="none")
    content_extraction_status: Mapped[str] = mapped_column(
        String(48), server_default="not_attempted"
    )
    # Pydantic HttpUrl permits up to 2083 characters.
    url: Mapped[str] = mapped_column(String(2083), unique=True)
    source: Mapped[str] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    category: Mapped[str] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(16))
    language_confidence: Mapped[float | None] = mapped_column(Float)
    language_status: Mapped[str] = mapped_column(String(32), server_default="not_attempted")
    language_detector: Mapped[str | None] = mapped_column(String(96))
    language_text_source: Mapped[str | None] = mapped_column(String(32))
    language_input_hash: Mapped[str | None] = mapped_column(String(64))
    topic_status: Mapped[str] = mapped_column(String(32), server_default="not_attempted")
    topic_model: Mapped[str | None] = mapped_column(String(160))
    topic_input_hash: Mapped[str | None] = mapped_column(String(64))
    topic_text_source: Mapped[str | None] = mapped_column(String(32))
    topic_language: Mapped[str | None] = mapped_column(String(16))
    topic_scores: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    article_topics: Mapped[list[ArticleTopicRecord]] = relationship(
        lazy="selectin",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by=lambda: (ArticleTopicRecord.confidence.desc(), ArticleTopicRecord.topic),
    )
    entity_status: Mapped[str] = mapped_column(String(32), server_default="not_attempted")
    entity_model: Mapped[str | None] = mapped_column(String(160))
    entity_input_hash: Mapped[str | None] = mapped_column(String(64))
    entity_text_source: Mapped[str | None] = mapped_column(String(32))
    entity_language: Mapped[str | None] = mapped_column(String(16))
    entity_text_truncated: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    entities: Mapped[list[EntityRecord]] = relationship(
        secondary=article_entities,
        lazy="selectin",
        viewonly=True,
        order_by=lambda: (EntityRecord.entity_type, EntityRecord.canonical_name),
    )
    keyword_status: Mapped[str] = mapped_column(String(32), server_default="not_attempted")
    keyword_model: Mapped[str | None] = mapped_column(String(160))
    keyword_input_hash: Mapped[str | None] = mapped_column(String(64))
    keyword_text_source: Mapped[str | None] = mapped_column(String(32))
    keyword_language: Mapped[str | None] = mapped_column(String(16))
    keyword_text_truncated: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    keywords: Mapped[list[ArticleKeywordRecord]] = relationship(
        lazy="selectin",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by=lambda: ArticleKeywordRecord.rank,
    )
    summary: Mapped[str | None] = mapped_column(Text)
    summary_one_sentence: Mapped[str | None] = mapped_column(Text)
    summary_bullets: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    summary_sentence_spans: Mapped[list[dict]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb")
    )
    summary_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary_status: Mapped[str] = mapped_column(String(32), server_default="not_attempted")
    summary_model: Mapped[str | None] = mapped_column(String(160))
    summary_input_hash: Mapped[str | None] = mapped_column(String(64))
    summary_text_source: Mapped[str | None] = mapped_column(String(32))
    summary_language: Mapped[str | None] = mapped_column(String(16))
    summary_source_chars: Mapped[int] = mapped_column(Integer, server_default="0")
    summary_text_truncated: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    rss_categories: Mapped[list[str]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb"), default=list
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    cleaning_version: Mapped[str | None] = mapped_column(String(16))
    cleaning_issues: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    processing_status: Mapped[str] = mapped_column(String(16), server_default="legacy")


Index(
    "ix_articles_lexical",
    func.newslens_lexical_vector(
        ArticleRecord.title, ArticleRecord.description, ArticleRecord.content
    ),
    postgresql_using="gin",
)
