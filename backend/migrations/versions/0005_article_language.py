"""Track language decisions separately from unknown/unprocessed language values.

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("articles", sa.Column("language_confidence", sa.Float(), nullable=True))
    op.add_column(
        "articles",
        sa.Column("language_status", sa.String(32), nullable=False, server_default="not_attempted"),
    )
    op.add_column("articles", sa.Column("language_detector", sa.String(96), nullable=True))
    op.add_column("articles", sa.Column("language_text_source", sa.String(32), nullable=True))
    op.add_column("articles", sa.Column("language_input_hash", sa.String(64), nullable=True))


def downgrade() -> None:
    for name in (
        "language_input_hash",
        "language_text_source",
        "language_detector",
        "language_status",
        "language_confidence",
    ):
        op.drop_column("articles", name)
    # The pre-existing language column and its values survive downgrade.
