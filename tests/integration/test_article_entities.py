import pytest
from alembic import command
from pydantic import HttpUrl
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from backend.app.database.session import create_session_factory
from backend.app.models.entity import EntityRecord, article_entities
from backend.app.services.article_cleaning import ArticleCleaningService
from backend.app.services.article_entities import ArticleEntityService
from backend.app.services.article_storage import ArticleStorageService
from ingestion.cleaners.processor import ArticleProcessor
from ingestion.parsers.rss import normalize_entry
from nlp.entities.extractor import EntityExtractor

pytestmark = pytest.mark.integration


class Backend:
    version = "entity-fixture/v1"
    languages = frozenset({"en"})
    available = True
    fail = False

    def predict(self, text, language):
        if self.fail:
            raise RuntimeError("Simulated model failure")
        if "hospital" in text:
            return [("London", "GPE")]
        return [("NVIDIA", "ORG"), ("nvidia", "ORG"), ("Jensen Huang", "PERSON"), ("US", "GPE")]


def setup(database, feed, retrieved_at):
    sessions = create_session_factory(database[0])
    backend = Backend()
    extractor = EntityExtractor(backend)
    storage = ArticleStorageService(sessions, entity_extractor=extractor)
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


def test_shared_normalized_entities_links_search_and_audit(database, feed, retrieved_at):
    sessions, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store(
        [article, article.model_copy(update={"url": HttpUrl("https://example.com/second")})]
    )
    records = storage.list_recent()
    assert len(records) == 2 and all(len(r.entities) == 3 for r in records)
    assert {e.id for e in records[0].entities} == {e.id for e in records[1].entities}
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(EntityRecord)) == 3
        assert session.scalar(select(func.count()).select_from(article_entities)) == 6
    assert len(storage.search_by_entity(" nViDiA ", entity_type="ORGANIZATION")) == 2
    assert len(storage.search_by_entity("United States", entity_type="COUNTRY")) == 2
    assert len(storage.search_by_entity("U.S.")) == 2
    assert len(storage.search_by_entity("NVIDIA", limit=1, offset=1)) == 1
    assert storage.search_by_entity("NVIDIA", entity_type="PERSON") == []
    assert storage.search_by_entity("%") == []
    audit = ArticleCleaningService(sessions).audit(article_id=records[0].id)[0]
    assert audit["clean"]["entity_status"] == "extracted"
    assert len(audit["clean"]["entities"]) == 3


def test_enrichment_replaces_links_using_retained_body_and_unchanged_duplicates_skip(
    database, feed, retrieved_at
):
    sessions, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    original = storage.list_recent()[0]
    enriched = ArticleProcessor().process(
        article,
        content=(
            "The hospital in London provides medical treatment for patients. "
            "Doctors opened a new healthcare center."
        ),
    )
    enriched.content_available = True
    storage.store_processed([enriched])
    stored = storage.get(original.id)
    assert [e.canonical_name for e in stored.entities] == ["london"]
    assert stored.entity_text_source == "title_content"
    assert storage.search_by_entity("NVIDIA") == []
    storage.store([article])
    assert storage.get(original.id).updated_at == stored.updated_at
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(article_entities)) == 1


def test_failed_refresh_retains_entities_and_provenance(database, feed, retrieved_at):
    sessions, backend, extractor, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    backend.fail = True
    assert ArticleEntityService(sessions, extractor).backfill_batch(refresh=True).errors == 1
    after = storage.get(before.id)
    assert after.entities == before.entities and after.entity_input_hash == before.entity_input_hash
    assert after.updated_at == before.updated_at
    assert (
        "entity_failure_previous_result_retained"
        in ArticleCleaningService(sessions).audit()[0]["issues"]
    )


def test_backfill_is_idempotent_and_refresh_is_explicit(database, feed, retrieved_at):
    sessions, _, extractor, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    with sessions.begin() as session:
        session.execute(
            text("UPDATE articles SET entity_status='not_attempted', entity_model=NULL")
        )
    service = ArticleEntityService(sessions, extractor)
    assert service.backfill_batch(1).extracted == 1
    assert service.backfill_batch(1).processed == 0
    assert service.backfill_batch(1, refresh=True).extracted == 1


def test_link_metadata_and_audit_are_atomic(database, feed, retrieved_at, monkeypatch):
    sessions, _, extractor, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    extractor.version = "replacement/v2"
    monkeypatch.setattr(
        "backend.app.repositories.processing.ProcessingRepository.add",
        lambda *a: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(RuntimeError):
        ArticleEntityService(sessions, extractor).backfill_batch(refresh=True)
    after = storage.get(before.id)
    assert after.entities == before.entities and after.entity_model == before.entity_model


def test_entity_constraints_and_article_cascade(database, feed, retrieved_at):
    sessions, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    with pytest.raises(DBAPIError) as exc, sessions.begin() as session:
        session.add(EntityRecord(name="wrong", canonical_name="wrong", entity_type="DATE"))
        session.flush()
    assert exc.value.orig.args[0]["C"] == "23514"
    with sessions.begin() as session:
        session.execute(text("DELETE FROM articles"))
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(article_entities)) == 0
        assert session.scalar(select(func.count()).select_from(EntityRecord)) == 3


def test_migration_roundtrip_preserves_previous_phase_data(database, feed, retrieved_at):
    _, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    engine, config, _ = database
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.check(config)
        command.downgrade(config, "0006")
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 1
        command.upgrade(config, "head")
    after = storage.get(before.id)
    assert after.language == before.language and after.article_topics == before.article_topics
    assert after.entities == [] and after.entity_status == "not_attempted"


def test_worker_extracts_after_content_and_language_and_saves_entities(database, feed, monkeypatch):
    from ingestion.http import RSSFetcher
    from ingestion.runner import build_update_service
    from nlp.entities.model import OnnxEntityBackend

    engine, _, settings = database
    body = "The hospital in London provides medical treatment for patients. " * 7
    rss = (
        '<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        "<channel><title>Feed</title><item><title>NVIDIA announces a new research center</title>"
        "<link>https://example.com/worker-story</link>"
        f"<content:encoded><![CDATA[<p>{body}</p>]]></content:encoded>"
        "</item></channel></rss>"
    ).encode()
    calls = []

    def predict(self, text, language):
        calls.append((text, language))
        return Backend().predict(text, language)

    monkeypatch.setattr(RSSFetcher, "fetch", lambda *_: rss)
    monkeypatch.setattr(OnnxEntityBackend, "predict", predict)
    outcome = build_update_service(settings, engine, [feed]).run()
    assert outcome.succeeded and outcome.report.inserted == 1
    assert len(calls) == 1 and calls[0][1] == "en"
    assert "NVIDIA" in calls[0][0] and "hospital" in calls[0][0]
    record = ArticleStorageService(create_session_factory(engine)).list_recent()[0]
    assert record.entity_status == "extracted" and record.entity_text_source == "title_content"
    assert [e.canonical_name for e in record.entities] == ["london"]
