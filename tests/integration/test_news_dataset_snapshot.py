from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import update

from backend.app.database.session import create_session_factory
from backend.app.evaluation.news_dataset import corpus_hash
from backend.app.evaluation.news_dataset_storage import compare_database, read_snapshot
from backend.app.models.article import ArticleRecord

pytestmark = pytest.mark.integration
NOW = datetime(2026, 9, 10, tzinfo=UTC)


def article(url, **changes):
    return ArticleRecord(
        title="Orion announced a factory opening",
        content="The factory will make solar panels.",
        url=url,
        source="Example",
        category="Business",
        language="en",
        published_at=NOW,
        retrieved_at=NOW,
        **{"cleaning_version": "1", "processing_status": "clean", **changes},
    )


def test_snapshot_reads_only_eligible_articles_and_detects_all_drift(database):
    engine, _, settings = database
    sessions = create_session_factory(engine)
    with sessions.begin() as session:
        eligible = article("https://example.com/one")
        rejected = article("https://example.com/rejected", processing_status="rejected")
        uncleaned = article("https://example.com/uncleaned", cleaning_version=None)
        session.add_all([eligible, rejected, uncleaned])
        session.flush()
        id_ = eligible.id
    frozen = read_snapshot(settings)
    assert [row.id for row in frozen] == [id_]
    dataset = SimpleNamespace(articles=frozen, corpus_sha256=corpus_hash(frozen))
    assert compare_database(dataset, settings)["matches"]
    with sessions.begin() as session:
        session.execute(
            update(ArticleRecord)
            .where(ArticleRecord.id == id_)
            .values(
                content="Updated content changes the frozen evidence.",
            )
        )
        new = article("https://example.com/new")
        session.add(new)
        session.flush()
        added_id = new.id
    result = compare_database(dataset, settings)
    assert not result["matches"]
    assert result["changed_article_ids"] == [id_]
    assert result["additional_article_ids"] == [added_id]
    assert result["missing_article_ids"] == []
    with sessions.begin() as session:
        session.delete(session.get(ArticleRecord, id_))
    result = compare_database(dataset, settings)
    assert result["missing_article_ids"] == [id_]


def test_snapshot_does_not_change_postgresql_rows(database):
    engine, _, settings = database
    sessions = create_session_factory(engine)
    with sessions.begin() as session:
        session.add(article("https://example.com/read-only"))
    first = read_snapshot(settings)
    second = read_snapshot(settings)
    assert first == second and corpus_hash(first) == corpus_hash(second)
