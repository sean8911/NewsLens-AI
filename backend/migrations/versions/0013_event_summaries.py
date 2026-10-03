"""Source-grounded event summary cache and provenance.

Revision ID: 0013
Revises: 0012
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "event_summaries",
        sa.Column(
            "event_id",
            sa.BigInteger(),
            sa.ForeignKey("events.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("policy_version", sa.String(64), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column(
            "payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("generated_at", sa.DateTime(timezone=True)),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error_type", sa.String(128)),
        sa.CheckConstraint(
            "status IN ('summarized', 'insufficient_sources', 'insufficient_text', "
            "'unsupported_language', 'stale', 'error')",
            name="status_valid",
        ),
        sa.CheckConstraint(
            "(status = 'summarized' AND generated_at IS NOT NULL AND payload <> '{}'::jsonb) "
            "OR (status <> 'summarized' AND generated_at IS NULL AND payload = '{}'::jsonb)",
            name="document_consistent",
        ),
    )


def downgrade():
    op.drop_table("event_summaries")
    # Phase 18 cannot validate the provenance of a Phase 19 summary.
    op.execute("UPDATE events SET summary = NULL")
