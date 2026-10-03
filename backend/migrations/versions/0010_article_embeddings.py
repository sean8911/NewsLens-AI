"""PostgreSQL ledger for rebuildable Qdrant embeddings.

Revision ID: 0010
Revises: 0009
"""

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "article_embeddings",
        sa.Column(
            "article_id",
            sa.BigInteger(),
            sa.ForeignKey("articles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("collection", sa.String(128), primary_key=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("input_hash", sa.String(64)),
        sa.Column("payload_hash", sa.String(64)),
        sa.Column("indexed_at", sa.DateTime(timezone=True)),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error_type", sa.String(128)),
        sa.CheckConstraint("status IN ('indexed', 'error')", name="status_valid"),
        sa.CheckConstraint("dimensions > 0", name="dimensions_positive"),
    )


def downgrade() -> None:
    op.drop_table("article_embeddings")
