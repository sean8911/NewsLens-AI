import pytest
from alembic import command
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from backend.app.database.session import create_session_factory
from backend.app.models.article_keyword import ArticleKeywordRecord
from backend.app.services.article_cleaning import ArticleCleaningService
from backend.app.services.article_keywords import ArticleKeywordService
from backend.app.services.article_storage import ArticleStorageService
from ingestion.cleaners.processor import ArticleProcessor
from ingestion.parsers.rss import normalize_entry
from nlp.keywords.extractor import KeywordExtractor

pytestmark = pytest.mark.integration


class Backend:
    version = "keyword-fixture/v1"
    languages = frozenset({"en"})
    fail = False
    empty = False

    def extract(self, text, language):
        if self.fail:
            raise RuntimeError("Simulated extraction failure")
        if self.empty:
            return []
        return (
            [("hospital", 0.02)]
            if "hospital" in text
            else [("computer chips", 0.02), ("NVIDIA", 0.1)]
        )


def setup(database, feed, retrieved_at):
    sessions = create_session_factory(database[0])
    backend = Backend()
    extractor = KeywordExtractor(backend)
    storage = ArticleStorageService(sessions, keyword_extractor=extractor)
    article = normalize_entry(
        {
            "title": "NVIDIA announces a new research center",
            "summary": (
                "The technology company announced that engineers will build new computer chips "
                "in the United States."
            ),
            "link": "https://example.com/story",
        },
        feed,
        retrieved_at,
    )
    return sessions, backend, extractor, storage, article


def test_ranked_keywords_and_provenance_are_stored_and_audited(database, feed, retrieved_at):
    sessions, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    record = storage.list_recent()[0]
    assert [(k.keyword, k.rank, k.score) for k in record.keywords] == [
        ("computer chips", 1, 0.02),
        ("NVIDIA", 2, 0.1),
    ]
    assert record.keyword_status == "extracted" and record.keyword_language == "en"
    audit = ArticleCleaningService(sessions).audit(article_id=record.id)[0]
    assert audit["clean"]["keywords"][0]["canonical_name"] == "computer chips"
    assert audit["clean"]["keyword_model"] == record.keyword_model


def test_enrichment_replaces_keywords_and_unchanged_duplicates_skip(database, feed, retrieved_at):
    _, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    enriched = ArticleProcessor().process(
        article,
        content=(
            "The hospital provides medical treatment and healthcare for patients. "
            "Doctors opened a new center."
        ),
    )
    enriched.content_available = True
    storage.store_processed([enriched])
    after = storage.get(before.id)
    assert [k.keyword for k in after.keywords] == ["hospital"]
    assert after.keyword_text_source == "title_content"
    storage.store([article])
    assert storage.get(before.id).updated_at == after.updated_at


@pytest.mark.parametrize("empty", [False, True])
def test_failure_preserves_prior_success_including_empty(database, feed, retrieved_at, empty):
    sessions, backend, extractor, storage, article = setup(database, feed, retrieved_at)
    backend.empty = empty
    storage.store([article])
    before = storage.list_recent()[0]
    backend.fail = True
    assert ArticleKeywordService(sessions, extractor).backfill_batch(refresh=True).errors == 1
    after = storage.get(before.id)
    assert after.keywords == before.keywords and after.keyword_status == before.keyword_status
    assert after.updated_at == before.updated_at
    assert (
        "keyword_failure_previous_result_retained"
        in ArticleCleaningService(sessions).audit()[0]["issues"]
    )


def test_backfill_idempotence_and_recovery_from_initial_failure(database, feed, retrieved_at):
    sessions, backend, extractor, storage, article = setup(database, feed, retrieved_at)
    backend.fail = True
    storage.store([article])
    assert storage.list_recent()[0].keyword_status == "error"
    backend.fail = False
    service = ArticleKeywordService(sessions, extractor)
    assert service.backfill_batch(1).extracted == 1
    assert service.backfill_batch(1).processed == 0
    assert service.backfill_batch(1, refresh=True).extracted == 1


def test_atomic_rollback_on_audit_failure(database, feed, retrieved_at, monkeypatch):
    sessions, _, extractor, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    extractor.version = "replacement/v2"
    monkeypatch.setattr(
        "backend.app.repositories.processing.ProcessingRepository.add",
        lambda *_: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(RuntimeError):
        ArticleKeywordService(sessions, extractor).backfill_batch(refresh=True)
    after = storage.get(before.id)
    assert after.keywords == before.keywords and after.keyword_model == before.keyword_model


@pytest.mark.parametrize("score", [-1, float("inf"), float("nan")])
def test_score_constraints(database, feed, retrieved_at, score):
    sessions, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    article_id = storage.list_recent()[0].id
    with pytest.raises(DBAPIError) as exc, sessions.begin() as session:
        session.add(
            ArticleKeywordRecord(
                article_id=article_id, keyword="bad", canonical_name="bad", rank=3, score=score
            )
        )
        session.flush()
    assert exc.value.orig.args[0]["C"] == "23514"


def test_migration_roundtrip_and_article_delete_cascade(database, feed, retrieved_at):
    sessions, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    engine, config, _ = database
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.check(config)
        command.downgrade(config, "0007")
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 1
        command.upgrade(config, "head")
    after = storage.get(before.id)
    assert after.entities == before.entities and after.article_topics == before.article_topics
    assert after.keywords == [] and after.keyword_status == "not_attempted"
    storage.store([article])
    with sessions.begin() as session:
        assert session.scalar(select(func.count()).select_from(ArticleKeywordRecord)) == 2
        session.execute(text("DELETE FROM articles"))
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(ArticleKeywordRecord)) == 0


def test_worker_extracts_keywords_once_after_content_and_language(database, feed, monkeypatch):
    from ingestion.http import RSSFetcher
    from ingestion.runner import build_update_service
    from nlp.keywords.extractor import YakeBackend

    engine, _, settings = database
    body = "The hospital provides healthcare and medical treatment for patients. " * 7
    rss = (
        '<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        "<channel><title>Feed</title><item><title>New research center opens</title>"
        "<link>https://example.com/worker-story</link>"
        f"<content:encoded><![CDATA[<p>{body}</p>]]></content:encoded>"
        "</item></channel></rss>"
    ).encode()
    calls = []

    def extract(self, text, language):
        calls.append((text, language))
        return [("hospital", 0.02)]

    monkeypatch.setattr(RSSFetcher, "fetch", lambda *_: rss)
    monkeypatch.setattr(YakeBackend, "extract", extract)
    outcome = build_update_service(settings, engine, [feed]).run()
    assert outcome.succeeded and len(calls) == 1
    assert calls[0][1] == "en" and "research center" in calls[0][0] and "hospital" in calls[0][0]
    record = ArticleStorageService(create_session_factory(engine)).list_recent()[0]
    assert record.keyword_text_source == "title_content"
    assert [k.keyword for k in record.keywords] == ["hospital"]
