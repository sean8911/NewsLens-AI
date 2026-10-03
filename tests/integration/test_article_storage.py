from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier

import pytest
from alembic import command
from sqlalchemy import inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.app.database.cli import main as read_main
from backend.app.database.session import create_session_factory
from backend.app.repositories.articles import ArticleRepository
from backend.app.schemas.article import Article
from backend.app.services.article_storage import ArticleStorageService
from ingestion.cleaners.processor import ArticleProcessor, ProcessingResult, snapshot
from ingestion.cli import main as collect_main
from ingestion.http import RSSFetcher
from ingestion.parsers.rss import parse_feed

pytestmark = pytest.mark.integration


def example(**changes):
    values = dict(
        title="Berita AI — 马来西亚",
        description=None,
        url="https://example.com/story",
        source="Example News",
        category="Malaysia",
        rss_categories=["AI", "Malaysia"],
        retrieved_at=datetime(2026, 9, 9, 8, tzinfo=UTC),
    )
    values.update(changes)
    return Article(**values)


def test_insert_read_and_duplicate_preserves_original(database):
    engine, _, _ = database
    service = ArticleStorageService(create_session_factory(engine))
    first = service.store([example()])
    second = service.store([example(title="Publisher changed the title")])
    assert (first.inserted, first.duplicates) == (1, 0)
    assert (second.inserted, second.duplicates) == (0, 1)
    record = service.list_recent()[0]
    assert record.title == "Berita AI — 马来西亚"
    assert record.published_at is None
    assert record.description is None
    assert record.content is None and record.language is None
    assert record.rss_categories == ["AI", "Malaysia"]
    assert record.created_at.tzinfo is not None
    assert record.updated_at == record.created_at
    assert service.get(record.id) == record
    assert service.get(record.id + 1000) is None
    with create_session_factory(engine)() as session:
        assert ArticleRepository(session).get_by_url(str(record.url)).id == record.id


def test_same_batch_duplicates_and_pagination(database):
    engine, _, _ = database
    service = ArticleStorageService(create_session_factory(engine))
    result = service.store([example(), example(), example(url="https://example.com/two")])
    assert (result.inserted, result.duplicates) == (2, 1)
    assert len(service.list_recent()) == 2
    first = service.list_recent(limit=1)[0]
    second = service.list_recent(limit=1, offset=1)[0]
    assert first.id > second.id  # Tie break for identical retrieval times.
    assert service.list_recent(offset=100) == []
    with pytest.raises(ValueError):
        service.list_recent(limit=101)


def test_database_constraint_rejects_direct_duplicate_and_batch_rolls_back(database):
    engine, _, _ = database
    sessions = create_session_factory(engine)
    service = ArticleStorageService(sessions)
    # A failure after the first insert must roll back that insert too.
    invalid = example(url="https://example.com/invalid").model_copy(update={"title": " "})
    with pytest.raises(DBAPIError) as exc:
        # Bypass the cleaner to exercise an actual database failure after an insert.
        service.store_processed(
            [
                ArticleProcessor().process(example()),
                ProcessingResult(snapshot({"test": "invalid"}), invalid),
            ]
        )
    assert exc.value.orig.args[0]["C"] == "23514"  # PostgreSQL check violation.
    assert service.list_recent() == []
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM article_processing_records")) == 0
    service.store([example()])
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text("""
            INSERT INTO articles (title, url, source, retrieved_at, category)
            SELECT title, url, source, retrieved_at, category FROM articles LIMIT 1
        """)
        )


def test_concurrent_collectors_insert_url_only_once(database):
    engine, _, _ = database
    sessions = create_session_factory(engine)
    barrier = Barrier(2)

    def store():
        barrier.wait(timeout=5)
        return ArticleStorageService(sessions).store([example()])

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(store) for _ in range(2)]
        results = [future.result(timeout=15) for future in futures]
    assert sum(result.inserted for result in results) == 1
    assert sum(result.duplicates for result in results) == 1


def test_migration_indexes_trigger_and_round_trip(database):
    engine, config, _ = database
    service = ArticleStorageService(create_session_factory(engine))
    service.store([example()])
    before = service.list_recent()[0]
    with engine.begin() as connection:
        connection.execute(text("UPDATE articles SET title = 'Updated'"))
    after = service.get(before.id)
    assert after.updated_at > before.updated_at
    assert after.created_at == before.created_at
    indexes = {index["name"] for index in inspect(engine).get_indexes("articles")}
    assert {
        "ix_articles_source",
        "ix_articles_category",
        "ix_articles_published_at",
        "ix_articles_retrieved_at_id",
        "uq_articles_url",
    } <= indexes
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.check(config)  # No mismatch between ORM metadata and migration.
        command.downgrade(config, "base")
        assert not inspect(connection).has_table("articles")
        command.upgrade(config, "head")
    assert service.list_recent() == []


def test_cli_store_then_read_and_repeat(database, monkeypatch, capsys, rss_bytes, caplog):
    _, _, settings = database
    monkeypatch.setenv("NEWSLENS_DATABASE_URL", settings.database_url.get_secret_value())
    monkeypatch.setenv("NEWSLENS_DATABASE_SCHEMA", settings.database_schema)
    monkeypatch.setattr(RSSFetcher, "fetch", lambda self, url: rss_bytes)
    assert collect_main(["--feed", "bbc-world", "--store"]) == 0
    capsys.readouterr()
    assert collect_main(["--feed", "bbc-world", "--store"]) == 0
    capsys.readouterr()
    assert read_main(["--limit", "100"]) == 0
    import json

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 3
    assert read_main(["--id", str(records[0]["id"])]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == records[0]["id"]
    assert read_main(["--id", "99999999"]) == 1


def test_unmigrated_database_fails_cleanly(database, monkeypatch, rss_bytes, caplog):
    engine, config, settings = database
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "base")
    monkeypatch.setenv("NEWSLENS_DATABASE_URL", settings.database_url.get_secret_value())
    monkeypatch.setenv("NEWSLENS_DATABASE_SCHEMA", settings.database_schema)
    monkeypatch.setattr(RSSFetcher, "fetch", lambda self, url: rss_bytes)
    assert collect_main(["--feed", "bbc-world", "--store"]) == 1
    assert "storage_failed" in caplog.text
    assert read_main([]) == 1
    assert "database_read_failed" in caplog.text
    assert settings.database_url.get_secret_value() not in caplog.text


def test_collected_batch_replay(database, feed, retrieved_at, rss_bytes):
    engine, _, _ = database
    articles = parse_feed(rss_bytes, feed, retrieved_at).articles
    service = ArticleStorageService(create_session_factory(engine))
    assert service.store(articles).inserted == 3
    assert service.store(articles).duplicates == 3
