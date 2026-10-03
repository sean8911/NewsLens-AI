from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


class EventSummaryRecord(Base):
    __tablename__ = "event_summaries"
    __table_args__ = (
        CheckConstraint(
            "status IN ('summarized', 'insufficient_sources', 'insufficient_text', "
            "'unsupported_language', 'stale', 'error')",
            name="status_valid",
        ),
        CheckConstraint(
            "(status = 'summarized' AND generated_at IS NOT NULL AND payload <> '{}'::jsonb) "
            "OR (status <> 'summarized' AND generated_at IS NULL AND payload = '{}'::jsonb)",
            name="document_consistent",
        ),
    )
    event_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("events.id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(32))
    policy_version: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(128))
