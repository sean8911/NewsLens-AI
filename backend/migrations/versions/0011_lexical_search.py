"""Automatically maintained PostgreSQL lexical search index.

Revision ID: 0011
Revises: 0010
"""

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Text-only arguments, fixed dictionary/filter, immutable text operations: this
    # wrapper is immutable even though generic jsonb_build_array accepts other types.
    # JSON string positions have a gap between fields, preventing cross-field phrases.
    op.execute("""
        CREATE FUNCTION newslens_lexical_vector(title text, description text, content text)
        RETURNS tsvector LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
            SELECT pg_catalog.jsonb_to_tsvector(
                'pg_catalog.simple'::regconfig,
                pg_catalog.jsonb_build_array(
                    left(coalesce(title, ''), 2000),
                    left(coalesce(description, ''), 8000),
                    left(coalesce(content, ''), 40000)
                ), '["string"]'::jsonb
            )
        $$
    """)
    op.execute("""
        CREATE INDEX ix_articles_lexical ON articles USING gin
        (newslens_lexical_vector(title, description, content))
    """)


def downgrade() -> None:
    op.drop_index("ix_articles_lexical", table_name="articles")
    op.execute("DROP FUNCTION newslens_lexical_vector(text, text, text)")
