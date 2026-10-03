"""Create initial article storage and indexes."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "articles",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("url", sa.String(2083), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("language", sa.String(16), nullable=True),
        sa.Column(
            "rss_categories",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_articles"),
        sa.UniqueConstraint("url", name="uq_articles_url"),
        sa.CheckConstraint("length(trim(title)) > 0", name=op.f("ck_articles_title_nonblank")),
    )
    op.create_index("ix_articles_source", "articles", ["source"])
    op.create_index("ix_articles_category", "articles", ["category"])
    op.create_index("ix_articles_published_at", "articles", ["published_at"])
    op.create_index("ix_articles_retrieved_at_id", "articles", ["retrieved_at", "id"])
    # Keep updated_at correct even for direct SQL updates, not just ORM writes.
    op.execute("""
        CREATE FUNCTION newslens_touch_article_updated_at() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            NEW.updated_at = clock_timestamp();
            RETURN NEW;
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER articles_touch_updated_at BEFORE UPDATE ON articles
        FOR EACH ROW EXECUTE FUNCTION newslens_touch_article_updated_at()
    """)


def downgrade() -> None:
    op.drop_table("articles")
    op.execute("DROP FUNCTION newslens_touch_article_updated_at()")
