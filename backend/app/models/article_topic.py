from sqlalchemy import BigInteger, CheckConstraint, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


class ArticleTopicRecord(Base):
    __tablename__ = "article_topics"
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
    )

    article_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True
    )
    topic: Mapped[str] = mapped_column(String(48), primary_key=True)
    confidence: Mapped[float] = mapped_column(Float)
