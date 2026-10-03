import json
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import func, select, text

from backend.app.database.session import create_session_factory
from backend.app.models.article import ArticleRecord
from backend.app.services.article_cleaning import ArticleCleaningService
from backend.app.services.article_language import ArticleLanguageService
from backend.app.services.article_storage import ArticleStorageService
from ingestion.cleaners.processor import ArticleProcessor
from ingestion.http import RSSFetcher
from ingestion.parsers.rss import normalize_entry
from ingestion.runner import build_update_service
from nlp.language.cli import main
from nlp.language.detector import LinguaBackend

pytestmark = pytest.mark.integration
CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures/language_cases.json").read_text(encoding="utf-8")
)


def article(feed, retrieved_at, case=0, url="https://example.com/story"):
    return normalize_entry(
        {"title": "News report", "summary": CASES[case]["text"], "link": url}, feed, retrieved_at
    )


def test_languages_are_stored_and_audited(database, feed, retrieved_at):
    sessions = create_session_factory(database[0])
    storage = ArticleStorageService(sessions)
    for i in [0, 2, 4, 6]:
        storage.store([article(feed, retrieved_at, i, f"https://example.com/{i}")])
    saved = {str(row.url): row for row in storage.list_recent()}
    for i in [0, 2, 4, 6]:
        record = saved[f"https://example.com/{i}"]
        assert record.language == CASES[i]["language"]
        assert record.language_detector.startswith("lingua-2.1.1-low/v1")
        assert record.language_text_source == "title_description"
        audit = ArticleCleaningService(sessions).audit(article_id=record.id)[0]
        assert audit["clean"]["language"] == record.language
        assert audit["clean"]["language_input_hash"] == record.language_input_hash


def test_duplicate_classifies_stored_body_and_preserves_identity(database, feed, retrieved_at):
    sessions = create_session_factory(database[0])
    storage = ArticleStorageService(sessions)
    original = article(feed, retrieved_at)
    storage.store([original])
    first = storage.list_recent()[0]
    incoming = ArticleProcessor().process(original, content=CASES[2]["text"])
    incoming.content_available = True
    incoming.content_extraction_method = "rss_content"
    storage.store_processed([incoming])
    enriched = storage.get(first.id)
    assert enriched.language == "ms" and enriched.language_text_source == "content"
    assert enriched.description == original.description and enriched.created_at == first.created_at
    # A later English feed description cannot relabel the retained Malay body.
    storage.store([original])
    after = storage.get(first.id)
    assert after.language == "ms" and after.updated_at == enriched.updated_at
    assert after.language_input_hash == enriched.language_input_hash


def test_unchanged_duplicate_does_not_rewrite_language(database, feed, retrieved_at):
    sessions = create_session_factory(database[0])
    storage = ArticleStorageService(sessions)
    original = article(feed, retrieved_at)
    storage.store([original])
    before = storage.list_recent()[0]
    storage.store(
        [article(feed, retrieved_at, 2)]
    )  # Publisher changes ignored by existing URL policy.
    after = storage.get(before.id)
    assert after.language == "en" and after.updated_at == before.updated_at
    assert len(ArticleCleaningService(sessions).audit(article_id=before.id)) == 2


def test_backfill_is_batched_resumable_and_idempotent(
    database, feed, retrieved_at, monkeypatch, capsys
):
    engine, _, settings = database
    sessions = create_session_factory(engine)
    storage = ArticleStorageService(sessions)
    for i in range(3):
        storage.store([article(feed, retrieved_at, i * 2, f"https://example.com/{i}")])
    with engine.begin() as c:
        c.execute(
            text(
                "UPDATE articles SET language=NULL, language_detector=NULL, "
                "language_status='not_attempted'"
            )
        )
    first = ArticleLanguageService(sessions).backfill_batch(1)
    assert first.processed == 1 and first.detected == 1
    monkeypatch.setenv("NEWSLENS_DATABASE_URL", settings.database_url.get_secret_value())
    monkeypatch.setenv("NEWSLENS_DATABASE_SCHEMA", settings.database_schema)
    assert main(["backfill", "--batch-size", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["processed"] == 2
    assert main(["backfill"]) == 0
    assert json.loads(capsys.readouterr().out)["processed"] == 0
    assert main(["backfill", "--refresh", "--batch-size", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["processed"] == 3
    assert {row.language for row in storage.list_recent()} == {"en", "ms", "zh"}


def test_detector_failure_does_not_break_worker_or_freshness(database, feed, monkeypatch):
    engine, _, settings = database
    rss = (
        f'<rss version="2.0"><channel><title>Feed</title><item><title>{CASES[0]["text"]}</title>'
        "<link>https://example.com/story</link></item></channel></rss>"
    ).encode()
    monkeypatch.setattr(RSSFetcher, "fetch", lambda *_: rss)
    monkeypatch.setattr(
        LinguaBackend, "rank", lambda *_: (_ for _ in ()).throw(RuntimeError("secret"))
    )
    result = build_update_service(settings, engine, [feed]).run()
    assert result.succeeded and result.report.inserted == 1
    saved = ArticleStorageService(create_session_factory(engine)).list_recent()[0]
    assert saved.language is None and saved.language_status == "error"


def test_backfill_rolls_back_label_and_audit_together(database, feed, retrieved_at, monkeypatch):
    from backend.app.repositories.processing import ProcessingRepository

    engine = database[0]
    sessions = create_session_factory(engine)
    storage = ArticleStorageService(sessions)
    storage.store([article(feed, retrieved_at)])
    with engine.begin() as c:
        c.execute(
            text(
                "UPDATE articles SET language=NULL, language_detector=NULL, "
                "language_status='not_attempted'"
            )
        )
    monkeypatch.setattr(
        ProcessingRepository, "add", lambda *_: (_ for _ in ()).throw(RuntimeError())
    )
    with pytest.raises(RuntimeError):
        ArticleLanguageService(sessions).backfill_batch()
    saved = storage.list_recent()[0]
    assert saved.language is None and saved.language_detector is None
    assert len(ArticleCleaningService(sessions).audit()) == 1


def test_migration_preserves_content_ids_and_language(database, feed, retrieved_at):
    engine, config, _ = database
    storage = ArticleStorageService(create_session_factory(engine))
    storage.store([article(feed, retrieved_at)])
    before = storage.list_recent()[0]
    with engine.begin() as c:
        config.attributes["connection"] = c
        command.check(config)
        command.downgrade(config, "0004")
        assert c.scalar(text("SELECT language FROM articles")) == "en"
        command.upgrade(config, "head")
    after = storage.get(before.id)
    assert after.language == "en" and after.description == before.description
    # Existing labels with unknown provenance are preserved by default.
    assert ArticleLanguageService(create_session_factory(engine)).backfill_batch().processed == 0
    with engine.connect() as c:
        assert c.scalar(select(func.count()).select_from(ArticleRecord)) == 1


def test_refresh_failure_preserves_previous_label_and_reports_error(
    database, feed, retrieved_at, monkeypatch, capsys
):
    engine, _, settings = database
    sessions = create_session_factory(engine)
    storage = ArticleStorageService(sessions)
    storage.store([article(feed, retrieved_at)])
    before = storage.list_recent()[0]
    monkeypatch.setenv("NEWSLENS_DATABASE_URL", settings.database_url.get_secret_value())
    monkeypatch.setenv("NEWSLENS_DATABASE_SCHEMA", settings.database_schema)
    monkeypatch.setattr(LinguaBackend, "rank", lambda *_: (_ for _ in ()).throw(RuntimeError()))
    assert main(["backfill", "--refresh"]) == 1
    totals = json.loads(capsys.readouterr().out)
    assert totals == {"processed": 1, "detected": 0, "unknown": 1, "errors": 1}
    after = storage.get(before.id)
    assert after.language == "en" and after.updated_at == before.updated_at
    audit = ArticleCleaningService(sessions).audit(article_id=before.id)[0]
    assert "language_error_previous_label_retained" in audit["issues"]


def test_worker_detects_after_content_extraction(database, feed, monkeypatch):
    engine, _, settings = database
    rss = (
        '<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        "<channel><title>Feed</title><item><title>An English news heading</title>"
        "<link>https://example.com/story</link>"
        f"<description>{CASES[0]['text']}</description>"
        f"<content:encoded><![CDATA[<p>{CASES[2]['text']}</p> "
        f"<p>{CASES[3]['text']}</p>]]></content:encoded></item></channel></rss>"
    ).encode()
    monkeypatch.setattr(RSSFetcher, "fetch", lambda *_: rss)
    outcome = build_update_service(settings, engine, [feed]).run()
    assert outcome.succeeded
    assert outcome.collection.records[0].language_result.language == "ms"
    stored = ArticleStorageService(create_session_factory(engine)).list_recent()[0]
    assert stored.content_available and stored.language == "ms"
    assert stored.language_text_source == "content"
