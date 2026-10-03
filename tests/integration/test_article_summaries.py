import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from backend.app.database.session import create_session_factory
from backend.app.services.article_cleaning import ArticleCleaningService
from backend.app.services.article_storage import ArticleStorageService
from backend.app.services.article_summaries import ArticleSummaryService
from ingestion.cleaners.processor import ArticleProcessor
from ingestion.parsers.rss import normalize_entry
from nlp.summarization.summarizer import ArticleSummarizer, SegtokBackend

pytestmark = pytest.mark.integration
BODY = (
    "Researchers said the trial did not prove a cure. "
    "The trial involved 1,200 adults across three hospitals. "
    "Doctors will continue monitoring the participants to assess long-term safety. "
    "The research team plans to publish the full results after the follow-up work has finished."
)


def setup(database, feed, retrieved_at):
    sessions = create_session_factory(database[0])
    summarizer = ArticleSummarizer()
    storage = ArticleStorageService(sessions, summarizer=summarizer)
    article = normalize_entry(
        {"title": "Trial research findings", "summary": BODY, "link": "https://example.com/story"},
        feed,
        retrieved_at,
    )
    return sessions, summarizer, storage, article


def test_summary_fields_grounding_and_audit(database, feed, retrieved_at):
    sessions, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    record = storage.list_recent()[0]
    assert record.summary_status == "summarized" and record.summary_created_at.tzinfo
    assert record.summary_one_sentence == "Researchers said the trial did not prove a cure."
    assert len(record.summary) < len(record.description)
    assert record.summary_text_source == "description"
    assert (
        " ".join(record.description[s["start"] : s["end"]] for s in record.summary_sentence_spans)
        == record.summary
    )
    audit = ArticleCleaningService(sessions).audit(article_id=record.id)[0]
    assert audit["clean"]["summary"] == record.summary
    assert audit["clean"]["summary_created_at"] == record.summary_created_at.isoformat()


def test_enrichment_refreshes_retained_body_and_duplicate_skips(database, feed, retrieved_at):
    _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    enriched = ArticleProcessor().process(
        article, content="The hospital opened a new research unit. " + BODY
    )
    enriched.content_available = True
    storage.store_processed([enriched])
    after = storage.get(before.id)
    assert after.summary_one_sentence == "The hospital opened a new research unit."
    assert after.summary_text_source == "content"
    assert after.summary_input_hash != before.summary_input_hash
    storage.store([article])
    again = storage.get(before.id)
    assert (
        again.summary_created_at == after.summary_created_at
        and again.updated_at == after.updated_at
    )


def test_failed_refresh_preserves_summary_and_timestamp(database, feed, retrieved_at, monkeypatch):
    sessions, summarizer, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    monkeypatch.setattr(SegtokBackend, "split", lambda *_: (_ for _ in ()).throw(RuntimeError()))
    assert ArticleSummaryService(sessions, summarizer).backfill_batch(refresh=True).errors == 1
    after = storage.get(before.id)
    assert after.summary == before.summary and after.summary_created_at == before.summary_created_at
    assert after.updated_at == before.updated_at
    assert (
        "summary_failure_previous_result_retained"
        in ArticleCleaningService(sessions).audit()[0]["issues"]
    )


def test_initial_failure_retains_article_and_backfill_retries(
    database, feed, retrieved_at, monkeypatch
):
    sessions, summarizer, storage, article = setup(database, feed, retrieved_at)
    with monkeypatch.context() as patch:
        patch.setattr(SegtokBackend, "split", lambda *_: (_ for _ in ()).throw(RuntimeError()))
        assert storage.store([article]).inserted == 1
    assert storage.list_recent()[0].summary_status == "error"
    service = ArticleSummaryService(sessions, summarizer)
    assert service.backfill_batch(1).summarized == 1
    assert service.backfill_batch(1).processed == 0
    assert service.backfill_batch(1, refresh=True).summarized == 1


def test_transaction_rollback_restores_summary_and_provenance(
    database, feed, retrieved_at, monkeypatch
):
    sessions, summarizer, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    summarizer.version = "replacement/v2"
    monkeypatch.setattr(
        "backend.app.repositories.processing.ProcessingRepository.add",
        lambda *_: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(RuntimeError):
        ArticleSummaryService(sessions, summarizer).backfill_batch(refresh=True)
    after = storage.get(before.id)
    assert after.summary == before.summary and after.summary_model == before.summary_model
    assert after.summary_created_at == before.summary_created_at


def test_migration_roundtrip_preserves_prior_nlp_and_summary_constraint(
    database, feed, retrieved_at
):
    sessions, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    with pytest.raises(DBAPIError) as exc, sessions.begin() as session:
        session.execute(text("UPDATE articles SET summary_source_chars=1"))
    assert exc.value.orig.args[0]["C"] == "23514"
    engine, config, _ = database
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.check(config)
        command.downgrade(config, "0008")
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 1
        command.upgrade(config, "head")
    after = storage.get(before.id)
    assert after.keywords == before.keywords and after.entities == before.entities
    assert after.summary_status == "not_attempted" and after.summary is None


def test_worker_summarizes_after_body_extraction_once(database, feed, monkeypatch):
    from ingestion.http import RSSFetcher
    from ingestion.runner import build_update_service

    engine, _, settings = database
    rss = (
        '<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        "<channel><title>Feed</title><item><title>New trial findings</title>"
        "<link>https://example.com/worker-story</link>"
        f"<content:encoded><![CDATA[<p>{BODY} {BODY}</p>]]></content:encoded>"
        "</item></channel></rss>"
    ).encode()
    calls = []
    original = SegtokBackend.split

    def split(self, text):
        calls.append(text)
        return original(self, text)

    monkeypatch.setattr(RSSFetcher, "fetch", lambda *_: rss)
    monkeypatch.setattr(SegtokBackend, "split", split)
    outcome = build_update_service(settings, engine, [feed]).run()
    assert outcome.succeeded and len(calls) == 1
    record = ArticleStorageService(create_session_factory(engine)).list_recent()[0]
    assert record.summary_status == "summarized" and record.summary_text_source == "content"
    assert record.summary_one_sentence in record.content
