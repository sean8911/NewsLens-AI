"""Add cleaning metadata and lossless pre-clean input snapshots."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("articles", sa.Column("cleaning_version", sa.String(16), nullable=True))
    op.add_column(
        "articles",
        sa.Column(
            "cleaning_issues",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "articles",
        sa.Column("processing_status", sa.String(16), server_default="legacy", nullable=False),
    )
    op.create_table(
        "article_processing_records",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("article_id", sa.BigInteger(), nullable=True),
        sa.Column("raw_json", sa.Text(), nullable=False),
        sa.Column("clean_json", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("issues", postgresql.JSONB(), nullable=False),
        sa.Column("cleaning_version", sa.String(16), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_article_processing_records"),
        sa.ForeignKeyConstraint(
            ["article_id"], ["articles.id"], ondelete="SET NULL", name="fk_processing_article"
        ),
    )
    op.create_index(
        "ix_article_processing_records_article_id", "article_processing_records", ["article_id"]
    )
    op.create_index(
        "ix_article_processing_records_status", "article_processing_records", ["status"]
    )


def downgrade() -> None:
    op.drop_table("article_processing_records")
    op.drop_column("articles", "processing_status")
    op.drop_column("articles", "cleaning_issues")
    op.drop_column("articles", "cleaning_version")
