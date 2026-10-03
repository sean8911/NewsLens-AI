from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Identity, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


class ProcessingRecord(Base):
    __tablename__ = "article_processing_records"
    __table_args__ = (
        Index("ix_article_processing_records_article_id", "article_id"),
        Index("ix_article_processing_records_status", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    article_id: Mapped[int | None] = mapped_column(
        ForeignKey("articles.id", name="fk_processing_article", ondelete="SET NULL")
    )
    raw_json: Mapped[str] = mapped_column(Text)
    clean_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    issues: Mapped[list[str]] = mapped_column(JSONB)
    cleaning_version: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
