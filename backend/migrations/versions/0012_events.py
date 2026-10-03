"""Event clusters, membership provenance and reconciliation audit.

Revision ID: 0012
Revises: 0011
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text()),
        sa.Column(
            "representative_article_id",
            sa.BigInteger(),
            sa.ForeignKey("articles.id", ondelete="SET NULL"),
        ),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("first_published_at", sa.DateTime(timezone=True)),
        sa.Column("last_published_at", sa.DateTime(timezone=True)),
        sa.Column("last_updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("policy_version", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.CheckConstraint("length(trim(title)) > 0", name="title_nonblank"),
    )
    op.create_index("ix_events_last_updated_at", "events", ["last_updated_at"])
    op.create_table(
        "article_events",
        sa.Column(
            "article_id",
            sa.BigInteger(),
            sa.ForeignKey("articles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "event_id",
            sa.BigInteger(),
            sa.ForeignKey("events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint(
            "similarity_score >= -1 AND similarity_score <= 1", name="similarity_valid"
        ),
    )
    op.create_index("ix_article_events_event_id", "article_events", ["event_id"])
    op.create_table(
        "event_cluster_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("report", postgresql.JSONB(), nullable=False),
    )


def downgrade():
    op.drop_table("event_cluster_runs")
    op.drop_table("article_events")
    op.drop_table("events")
