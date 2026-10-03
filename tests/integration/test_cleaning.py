import json

import pytest
from alembic import command
from sqlalchemy import text

from backend.app.database.session import create_session_factory
from backend.app.schemas.article import Article
from backend.app.services.article_cleaning import ArticleCleaningService
from backend.app.services.article_storage import ArticleStorageService, PendingCleaningError
from ingestion.cleaners.cli import main
from ingestion.cleaners.processor import ProcessingResult, snapshot
from ingestion.parsers.rss import parse_feed

pytestmark = pytest.mark.integration


def legacy(engine, title="<b>Legacy news</b>", url="https://example.com/legacy?utm_source=rss"):
    with engine.begin() as connection:
        return connection.scalar(
            text("""
            INSERT INTO articles (title, url, source, description, content, retrieved_at, category)
            VALUES (:title, :url, 'BBC', '<p>Legacy description</p>', '<div>Old content</div>',
                    '2026-09-09T08:00:00Z', 'Technology') RETURNING id
        """),
            {"title": title, "url": url},
        )


def test_feed_inputs_saved_for_accepted_and_rejected_entries(
    database, feed, rss_bytes, retrieved_at
):
    engine, _, _ = database
    sessions = create_session_factory(engine)
    parsed = parse_feed(rss_bytes, feed, retrieved_at)
    result = ArticleStorageService(sessions).store_processed(parsed.records)
    assert (result.inserted, result.rejected) == (3, 2)
    audit = ArticleCleaningService(sessions).audit(limit=100)
    assert len(audit) == 5
    rejected = [item for item in audit if item["status"] == "rejected"]
    assert len(rejected) == 2
    assert all(item["raw"]["entry"] and item["issues"] for item in rejected)
    assert all(item["article_id"] is None for item in rejected)
    stored = ArticleStorageService(sessions).list_recent()
    assert any(record.description == "Research & technology update." for record in stored)
    assert all(record.cleaning_version == "1" for record in stored)


def test_tracking_variants_deduplicate_and_preserve_both_inputs(database, retrieved_at):
    engine, _, _ = database
    sessions = create_session_factory(engine)
    first = Article(
        title="AI update",
        url="https://example.com/news?utm_source=one",
        source="BBC",
        category="AI",
        retrieved_at=retrieved_at,
    )
    second = Article.model_validate(
        first.model_dump() | {"url": "https://example.com/news?utm_source=two"}
    )
    result = ArticleStorageService(sessions).store([first, second])
    assert (result.inserted, result.duplicates) == (1, 1)
    audit = ArticleCleaningService(sessions).audit()
    assert {item["raw"]["article"]["url"] for item in audit} == {str(first.url), str(second.url)}
    assert len({item["article_id"] for item in audit}) == 1


def test_raw_snapshot_retains_nul_and_surrogate(database, retrieved_at):
    engine, _, _ = database
    sessions = create_session_factory(engine)
    article = Article(
        title="A\x00B",
        url="https://example.com/unicode",
        source="News",
        category="Technology",
        retrieved_at=retrieved_at,
    )
    ArticleStorageService(sessions).store([article])
    audit = ArticleCleaningService(sessions).audit()[0]
    assert audit["raw"]["article"]["title"] == article.title
    assert "\x00" not in audit["clean"]["article"]["title"]
    rejected = ProcessingResult(
        snapshot({"entry": {"title": "bad\ud800"}}), None, ["normalization_failed"]
    )
    ArticleStorageService(sessions).store_processed([rejected])
    assert ArticleCleaningService(sessions).audit(status="rejected")[0]["raw"]["entry"][
        "title"
    ] == ("bad\ud800")


def test_migration_and_backfill_preserve_ids_and_original_content(database):
    engine, config, _ = database
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "0001")
    article_id = legacy(engine)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    sessions = create_session_factory(engine)
    service = ArticleCleaningService(sessions)
    result = service.backfill_batch()
    assert result.processed == result.cleaned == 1
    record = ArticleStorageService(sessions).get(article_id)
    assert record.title == "Legacy news"
    assert record.source == "BBC News"
    assert record.content == "Old content"
    assert str(record.url) == "https://example.com/legacy"
    raw = service.audit(article_id=article_id)[0]["raw"]["record"]
    assert raw["title"] == "<b>Legacy news</b>"
    assert raw["content"] == "<div>Old content</div>"
    assert raw["id"] == article_id
    from datetime import datetime

    assert record.created_at == datetime.fromisoformat(raw["created_at"])
    assert service.backfill_batch().processed == 0


def test_backfill_collision_keeps_both_ids_and_flags_review(database):
    engine, _, _ = database
    first = legacy(engine, url="https://example.com/legacy")
    second = legacy(engine)
    sessions = create_session_factory(engine)
    service = ArticleCleaningService(sessions)
    result = service.backfill_batch()
    assert result.processed == 2 and result.url_collisions == 1
    records = ArticleStorageService(sessions).list_recent()
    assert {record.id for record in records} == {first, second}
    collision = ArticleStorageService(sessions).get(second)
    assert collision.processing_status == "review"
    assert "utm_source" in str(collision.url)
    audit = service.audit(status="url_collision")[0]
    assert audit["clean"]["article"]["url"] == "https://example.com/legacy"


def test_rejected_legacy_article_is_retained_and_excluded_from_lists(database):
    engine, _, _ = database
    article_id = legacy(engine, title="<script>bad()</script>")
    sessions = create_session_factory(engine)
    cleaning = ArticleCleaningService(sessions)
    assert cleaning.backfill_batch().rejected == 1
    storage = ArticleStorageService(sessions)
    assert storage.list_recent() == []
    assert storage.get(article_id) is None
    assert cleaning.audit(article_id=article_id)[0]["raw"]["record"]["title"].startswith("<script>")


def test_cleaning_cli_backfill_is_resumable_and_audit_is_readable(database, monkeypatch, capsys):
    engine, _, settings = database
    legacy(engine)
    legacy(engine, url="https://example.com/second")
    monkeypatch.setenv("NEWSLENS_DATABASE_URL", settings.database_url.get_secret_value())
    monkeypatch.setenv("NEWSLENS_DATABASE_SCHEMA", settings.database_schema)
    assert main(["backfill", "--batch-size", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["processed"] == 2
    assert main(["backfill"]) == 0
    assert json.loads(capsys.readouterr().out)["processed"] == 0
    assert main(["audit", "--status", "backfilled", "--limit", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["raw"]["format"] == "stored_article"


def test_legacy_invalid_url_is_preserved_for_inspection(database):
    engine, _, _ = database
    article_id = legacy(engine, url="not a URL")
    sessions = create_session_factory(engine)
    result = ArticleCleaningService(sessions).backfill_batch()
    assert result.rejected == 1
    assert ArticleStorageService(sessions).get(article_id) is None
    audit = ArticleCleaningService(sessions).audit(article_id=article_id)[0]
    assert audit["raw"]["record"]["url"] == "not a URL"
    assert audit["issues"] == ["legacy_validation_failed"]


def test_storage_requires_backfill_to_avoid_parallel_raw_and_clean_urls(database):
    engine, _, _ = database
    legacy(engine)
    sessions = create_session_factory(engine)
    with pytest.raises(PendingCleaningError):
        ArticleStorageService(sessions).store([])
