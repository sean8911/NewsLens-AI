"""Explicit, read-only comparison of the portable dataset with PostgreSQL."""

from sqlalchemy import select, text

from backend.app.database.session import create_database_engine
from backend.app.evaluation.news_dataset import SnapshotArticle, corpus_hash
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchRequest
from backend.app.services.search_filters import apply_article_filters


def read_snapshot(settings) -> list[SnapshotArticle]:
    engine = create_database_engine(settings)
    fields = tuple(SnapshotArticle.model_fields)
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                connection.execute(text("SET TRANSACTION READ ONLY"))
                statement = (
                    apply_article_filters(
                        select(*(getattr(ArticleRecord, name) for name in fields)),
                        SearchRequest(q="evaluation snapshot"),
                    )
                    .order_by(ArticleRecord.id)
                    .limit(5001)
                )
                rows = connection.execute(statement).mappings().all()
                if len(rows) > 5000:
                    raise ValueError("Snapshot exceeds the 5,000-article limit")
                return [SnapshotArticle.model_validate(dict(row)) for row in rows]
    finally:
        engine.dispose()


def compare_database(dataset, settings) -> dict:
    current = {article.id: article for article in read_snapshot(settings)}
    frozen = {article.id: article for article in dataset.articles}
    changed = [id_ for id_ in sorted(current.keys() & frozen.keys()) if current[id_] != frozen[id_]]
    return {
        "matches": corpus_hash(list(current.values())) == dataset.corpus_sha256,
        "current_article_count": len(current),
        "frozen_article_count": len(frozen),
        "missing_article_ids": sorted(frozen.keys() - current.keys()),
        "additional_article_ids": sorted(current.keys() - frozen.keys()),
        "changed_article_ids": changed,
        "current_corpus_sha256": corpus_hash(list(current.values())),
        "frozen_corpus_sha256": dataset.corpus_sha256,
    }
