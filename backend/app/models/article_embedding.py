from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


class ArticleEmbeddingRecord(Base):
    """Synchronization ledger only. Vector values belong in Qdrant."""

    __tablename__ = "article_embeddings"
    __table_args__ = (
        CheckConstraint("status IN ('indexed', 'error')", name="status_valid"),
        CheckConstraint("dimensions > 0", name="dimensions_positive"),
    )
    article_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True
    )
    collection: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(64))
    dimensions: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    input_hash: Mapped[str | None] = mapped_column(String(64))
    payload_hash: Mapped[str | None] = mapped_column(String(64))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    error_type: Mapped[str | None] = mapped_column(String(128))
