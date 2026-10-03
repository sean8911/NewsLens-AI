"""Store multi-label topic scores and classifier provenance.

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "articles",
        sa.Column("topic_status", sa.String(32), nullable=False, server_default="not_attempted"),
    )
    for name, length in [
        ("topic_model", 160),
        ("topic_input_hash", 64),
        ("topic_text_source", 32),
        ("topic_language", 16),
    ]:
        op.add_column("articles", sa.Column(name, sa.String(length), nullable=True))
    op.add_column(
        "articles",
        sa.Column("topic_scores", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_table(
        "article_topics",
        sa.Column(
            "article_id",
            sa.BigInteger(),
            sa.ForeignKey("articles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("topic", sa.String(48), primary_key=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
    )


def downgrade() -> None:
    op.drop_table("article_topics")
    for name in (
        "topic_scores",
        "topic_language",
        "topic_text_source",
        "topic_input_hash",
        "topic_model",
        "topic_status",
    ):
        op.drop_column("articles", name)
