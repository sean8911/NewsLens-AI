from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


class ArticleKeywordRecord(Base):
    __tablename__ = "article_keywords"
    __table_args__ = (
        CheckConstraint("score >= 0 AND score < 'Infinity'::float8", name="score_valid"),
        CheckConstraint("rank >= 1 AND rank <= 10", name="rank_valid"),
        CheckConstraint(
            "length(trim(keyword)) > 0 AND length(trim(canonical_name)) > 0",
            name="keyword_nonblank",
        ),
        UniqueConstraint("article_id", "rank", name="uq_article_keywords_rank"),
        Index("ix_article_keywords_canonical_name", "canonical_name", "article_id"),
    )
    article_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True
    )
    canonical_name: Mapped[str] = mapped_column(String(256), primary_key=True)
    keyword: Mapped[str] = mapped_column(String(256))
    score: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)
