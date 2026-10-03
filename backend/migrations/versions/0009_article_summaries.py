"""Grounded article summaries and extraction provenance.

Revision ID: 0009
Revises: 0008
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "articles",
        sa.Column("summary_status", sa.String(32), nullable=False, server_default="not_attempted"),
    )
    for name, length in [
        ("summary_model", 160),
        ("summary_input_hash", 64),
        ("summary_text_source", 32),
        ("summary_language", 16),
    ]:
        op.add_column("articles", sa.Column(name, sa.String(length), nullable=True))
    for name in ("summary", "summary_one_sentence"):
        op.add_column("articles", sa.Column(name, sa.Text(), nullable=True))
    for name in ("summary_bullets", "summary_sentence_spans"):
        op.add_column(
            "articles",
            sa.Column(name, JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        )
    op.add_column(
        "articles",
        sa.Column("summary_source_chars", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "articles",
        sa.Column(
            "summary_text_truncated", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
    )
    op.add_column(
        "articles", sa.Column("summary_created_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_check_constraint(
        "summary_consistent",
        "articles",
        "(summary IS NULL AND summary_one_sentence IS NULL AND summary_created_at IS NULL "
        "AND summary_status <> 'summarized') OR "
        "(summary IS NOT NULL AND summary_one_sentence IS NOT NULL "
        "AND summary_created_at IS NOT NULL "
        "AND summary_status = 'summarized' AND length(summary) > 0 "
        "AND length(summary_one_sentence) > 0 AND length(summary) < summary_source_chars "
        "AND length(summary_one_sentence) <= length(summary))",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_articles_summary_consistent"), "articles", type_="check")
    for name in (
        "summary_created_at",
        "summary_text_truncated",
        "summary_source_chars",
        "summary_sentence_spans",
        "summary_bullets",
        "summary_one_sentence",
        "summary",
        "summary_language",
        "summary_text_source",
        "summary_input_hash",
        "summary_model",
        "summary_status",
    ):
        op.drop_column("articles", name)
