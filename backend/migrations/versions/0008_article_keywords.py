"""Ranked article keywords and extraction provenance.

Revision ID: 0008
Revises: 0007
"""

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "articles",
        sa.Column("keyword_status", sa.String(32), nullable=False, server_default="not_attempted"),
    )
    for name, length in [
        ("keyword_model", 160),
        ("keyword_input_hash", 64),
        ("keyword_text_source", 32),
        ("keyword_language", 16),
    ]:
        op.add_column("articles", sa.Column(name, sa.String(length), nullable=True))
    op.add_column(
        "articles",
        sa.Column(
            "keyword_text_truncated", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
    )
    op.create_table(
        "article_keywords",
        sa.Column(
            "article_id",
            sa.BigInteger(),
            sa.ForeignKey("articles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("canonical_name", sa.String(256), primary_key=True),
        sa.Column("keyword", sa.String(256), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.CheckConstraint("score >= 0 AND score < 'Infinity'::float8", name="score_valid"),
        sa.CheckConstraint("rank >= 1 AND rank <= 10", name="rank_valid"),
        sa.CheckConstraint(
            "length(trim(keyword)) > 0 AND length(trim(canonical_name)) > 0",
            name="keyword_nonblank",
        ),
        sa.UniqueConstraint("article_id", "rank", name="uq_article_keywords_rank"),
    )
    op.create_index(
        "ix_article_keywords_canonical_name", "article_keywords", ["canonical_name", "article_id"]
    )


def downgrade() -> None:
    op.drop_table("article_keywords")
    for name in (
        "keyword_text_truncated",
        "keyword_language",
        "keyword_text_source",
        "keyword_input_hash",
        "keyword_model",
        "keyword_status",
    ):
        op.drop_column("articles", name)
