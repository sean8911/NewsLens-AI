from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    ForeignKey,
    Identity,
    Index,
    String,
    Table,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base

article_entities = Table(
    "article_entities",
    Base.metadata,
    Column(
        "article_id", BigInteger, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True
    ),
    Column(
        "entity_id", BigInteger, ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    ),
    Index("ix_article_entities_entity_id", "entity_id", "article_id"),
)


class EntityRecord(Base):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("entity_type", "canonical_name", name="uq_entities_type_name"),
        CheckConstraint("length(trim(name)) > 0", name="name_nonblank"),
        CheckConstraint("length(trim(canonical_name)) > 0", name="canonical_nonblank"),
        CheckConstraint(
            "entity_type IN ('PERSON','ORGANIZATION','LOCATION','COUNTRY','PRODUCT')",
            name="valid_type",
        ),
        Index("ix_entities_canonical_name", "canonical_name"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(String(512))
    canonical_name: Mapped[str] = mapped_column(String(512))
    entity_type: Mapped[str] = mapped_column(String(16))
