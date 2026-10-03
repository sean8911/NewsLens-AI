from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Identity, Index, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"
    __table_args__ = (Index("ix_ingestion_runs_status_finished_at", "status", "finished_at"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), server_default="running")
    feeds_checked: Mapped[int] = mapped_column(Integer)
    successful_feeds: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    failed_feeds: Mapped[dict[str, str]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    articles_found: Mapped[int] = mapped_column(Integer, server_default="0")
    articles_accepted: Mapped[int] = mapped_column(Integer, server_default="0")
    articles_changed: Mapped[int] = mapped_column(Integer, server_default="0")
    inserted: Mapped[int] = mapped_column(Integer, server_default="0")
    duplicates: Mapped[int] = mapped_column(Integer, server_default="0")
    rejected: Mapped[int] = mapped_column(Integer, server_default="0")
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    error_type: Mapped[str | None] = mapped_column(String(128))
