"""Record article content availability and extraction provenance.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "articles",
        sa.Column(
            "content_available", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
    )
    op.add_column(
        "articles",
        sa.Column(
            "content_extraction_method", sa.String(32), nullable=False, server_default="none"
        ),
    )
    op.add_column(
        "articles",
        sa.Column(
            "content_extraction_status",
            sa.String(48),
            nullable=False,
            server_default="not_attempted",
        ),
    )
    # Keep legacy content verbatim; do not retroactively claim it was extracted or verified.


def downgrade() -> None:
    op.drop_column("articles", "content_extraction_status")
    op.drop_column("articles", "content_extraction_method")
    op.drop_column("articles", "content_available")
