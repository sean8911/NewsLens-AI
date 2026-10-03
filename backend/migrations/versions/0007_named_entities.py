"""Normalized named entities and article links.

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "articles",
        sa.Column("entity_status", sa.String(32), nullable=False, server_default="not_attempted"),
    )
    for name, length in [
        ("entity_model", 160),
        ("entity_input_hash", 64),
        ("entity_text_source", 32),
        ("entity_language", 16),
    ]:
        op.add_column("articles", sa.Column(name, sa.String(length), nullable=True))
    op.add_column(
        "articles",
        sa.Column(
            "entity_text_truncated", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
    )
    op.create_table(
        "entities",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("canonical_name", sa.String(512), nullable=False),
        sa.Column("entity_type", sa.String(16), nullable=False),
        sa.UniqueConstraint("entity_type", "canonical_name", name="uq_entities_type_name"),
        sa.CheckConstraint("length(trim(name)) > 0", name="name_nonblank"),
        sa.CheckConstraint("length(trim(canonical_name)) > 0", name="canonical_nonblank"),
        sa.CheckConstraint(
            "entity_type IN ('PERSON','ORGANIZATION','LOCATION','COUNTRY','PRODUCT')",
            name="valid_type",
        ),
    )
    op.create_index("ix_entities_canonical_name", "entities", ["canonical_name"])
    op.create_table(
        "article_entities",
        sa.Column(
            "article_id",
            sa.BigInteger(),
            sa.ForeignKey("articles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "entity_id",
            sa.BigInteger(),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    op.create_index(
        "ix_article_entities_entity_id", "article_entities", ["entity_id", "article_id"]
    )


def downgrade() -> None:
    op.drop_table("article_entities")
    op.drop_table("entities")
    for name in (
        "entity_text_truncated",
        "entity_language",
        "entity_text_source",
        "entity_input_hash",
        "entity_model",
        "entity_status",
    ):
        op.drop_column("articles", name)
