import json
from copy import deepcopy

import pytest
from alembic import command
from sqlalchemy import inspect, text

from backend.app.database.session import create_session_factory
from backend.app.services.article_cleaning import ArticleCleaningService
from backend.app.services.article_storage import ArticleStorageService
from ingestion.extraction.http import ArticleFetcher
from ingestion.extraction.policy import SourcePolicy
from ingestion.http import RSSFetcher
from ingestion.parsers.rss import parse_feed
from ingestion.runner import build_update_service

pytestmark = pytest.mark.integration
BODY = (
    "Scientists released the results of a detailed study involving several research teams. "
    "They compared experimental observations with earlier predictions and published their "
    "methods for independent review. The findings offer useful evidence for future studies "
    "and describe the limitations of the available data. More measurements are planned next year."
)
RSS = (
    f'<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">'
    "<channel><title>Example</title><item><title>A research study</title>"
    "<link>https://example.com/study</link><description>A short summary.</description>"
    f"<content:encoded><![CDATA[<p>{BODY}</p>]]></content:encoded>"
    "</item></channel></rss>"
).encode()


def test_rss_content_persists_with_audit_and_duplicate_protection(database, feed, retrieved_at):
    engine, _, _ = database
    sessions = create_session_factory(engine)
    storage = ArticleStorageService(sessions)
    record = parse_feed(RSS, feed, retrieved_at).records[0]
    assert storage.store_processed([record]).inserted == 1
    assert storage.store_processed([record]).duplicates == 1
    saved = storage.list_recent()[0]
    assert saved.content == BODY and saved.content_available
    assert saved.content_extraction_method == "rss_content"
    assert saved.description == "A short summary."
    audit = ArticleCleaningService(sessions).audit(article_id=saved.id)
    assert audit[0]["clean"]["content"] == BODY
    assert audit[0]["clean"]["content_extraction_method"] == "rss_content"
    assert "content" in audit[0]["raw"]["entry"]


def test_existing_summary_gains_content_without_changing_identity(database, feed, retrieved_at):
    engine, _, _ = database
    storage = ArticleStorageService(create_session_factory(engine))
    full = parse_feed(RSS, feed, retrieved_at).records[0]
    assert storage.store([full.article]).inserted == 1
    original = storage.list_recent()[0]
    assert original.content is None
    assert storage.store_processed([full]).duplicates == 1
    saved = storage.get(original.id)
    assert saved.content == BODY and saved.content_available
    assert saved.created_at == original.created_at and saved.description == original.description
    assert saved.updated_at >= original.updated_at
    # A failed or lower-quality retry cannot downgrade the stored body.
    failed = deepcopy(full)
    failed.content = None
    failed.content_available = False
    failed.content_extraction_status = "extraction_error"
    storage.store_processed([failed])
    assert storage.get(original.id).content == BODY


def test_content_enrichment_rolls_back_with_caller_transaction(database, feed, retrieved_at):
    engine, _, _ = database
    sessions = create_session_factory(engine)
    storage = ArticleStorageService(sessions)
    full = parse_feed(RSS, feed, retrieved_at).records[0]
    storage.store([full.article])
    with pytest.raises(RuntimeError), sessions.begin() as session:
        storage.store_in_session(session, [full])
        session.flush()
        raise RuntimeError("simulated downstream failure")
    assert storage.list_recent()[0].content is None
    assert len(ArticleCleaningService(sessions).audit()) == 1


def test_worker_extracts_web_body_then_avoids_refetch(database, feed, monkeypatch, tmp_path):
    engine, _, settings = database
    settings.content_web_enabled = True
    path = tmp_path / "policies.json"
    source = SourcePolicy(
        host="example.com",
        terms_allow_extraction=True,
        terms_url="https://example.com/terms",
        review_note="Fixture only",
    )
    path.write_text(json.dumps([source.model_dump()]))
    settings.content_policies_path = path
    rss = RSS.replace(BODY.encode(), b"Too short")
    monkeypatch.setattr(RSSFetcher, "fetch", lambda *_: rss)
    monkeypatch.setattr(ArticleFetcher, "validate", lambda *_: ("example.com", source))
    calls = []

    def fetch(_self, url):
        calls.append(url)
        return (
            f'<script type="application/ld+json">'
            f"{json.dumps({'@type': 'NewsArticle', 'articleBody': BODY})}</script>"
        ).encode()

    monkeypatch.setattr(ArticleFetcher, "fetch", fetch)
    service = build_update_service(settings, engine, [feed])
    first, second = service.run(), service.run()
    assert first.succeeded and second.succeeded
    assert first.report.inserted == 1 and second.report.duplicates == 1
    assert calls == ["https://example.com/study"]
    saved = ArticleStorageService(create_session_factory(engine)).list_recent()[0]
    assert saved.content == BODY and saved.content_extraction_method == "json_ld"


def test_extraction_failure_keeps_ingestion_successful(database, feed, monkeypatch):
    engine, _, settings = database
    settings.content_web_enabled = True
    monkeypatch.setattr(RSSFetcher, "fetch", lambda *_: RSS.replace(BODY.encode(), b"Short"))
    monkeypatch.setattr(
        ArticleFetcher,
        "validate",
        lambda *_: (_ for _ in ()).throw(RuntimeError("simulated adapter failure")),
    )
    result = build_update_service(settings, engine, [feed]).run()
    assert result.succeeded and result.report.inserted == 1
    saved = ArticleStorageService(create_session_factory(engine)).list_recent()[0]
    assert saved.content is None and not saved.content_available
    assert saved.content_extraction_status == "extraction_error"
    assert saved.description == "A short summary."


def test_content_migration_roundtrip_preserves_legacy_content(database, feed, retrieved_at):
    engine, config, _ = database
    storage = ArticleStorageService(create_session_factory(engine))
    record = parse_feed(RSS, feed, retrieved_at).records[0]
    storage.store_processed([record])
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.check(config)
        command.downgrade(config, "0003")
        assert "content_available" not in {
            col["name"] for col in inspect(connection).get_columns("articles")
        }
        assert connection.scalar(text("SELECT content FROM articles")) == BODY
        command.upgrade(config, "head")
    saved = storage.list_recent()[0]
    assert saved.content == BODY
    assert not saved.content_available  # No invented provenance for legacy body text.
    changed = deepcopy(record)
    changed.content = "A different body"
    storage.store_processed([changed])
    assert storage.get(saved.id).content == BODY
