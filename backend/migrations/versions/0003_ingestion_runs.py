"""Persist update outcomes and freshness across worker restarts."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(16), server_default="running", nullable=False),
        sa.Column("feeds_checked", sa.Integer(), nullable=False),
        sa.Column(
            "successful_feeds",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "failed_feeds",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("articles_found", sa.Integer(), server_default="0", nullable=False),
        sa.Column("articles_accepted", sa.Integer(), server_default="0", nullable=False),
        sa.Column("articles_changed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("inserted", sa.Integer(), server_default="0", nullable=False),
        sa.Column("duplicates", sa.Integer(), server_default="0", nullable=False),
        sa.Column("rejected", sa.Integer(), server_default="0", nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("error_type", sa.String(128), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_ingestion_runs"),
    )
    op.create_index(
        "ix_ingestion_runs_status_finished_at", "ingestion_runs", ["status", "finished_at"]
    )


def downgrade() -> None:
    op.drop_table("ingestion_runs")
