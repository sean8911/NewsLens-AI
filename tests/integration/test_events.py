from datetime import UTC, datetime, timedelta

import pytest
from alembic import command
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import func, select, text

from backend.app.database.session import create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.models.event import ArticleEventRecord, EventClusterRun, EventRecord
from backend.app.schemas.event import EventListRequest
from backend.app.services.article_embeddings import ArticleEmbeddingService
from backend.app.services.event_clustering import EventClusteringService
from backend.app.services.events import EventReadService
from backend.app.services.news_updates import NewsUpdateService
from ingestion.collector import RSSCollector
from nlp.embeddings.store import QdrantEmbeddingStore
from nlp.events.clustering import ClusterConfig

pytestmark = pytest.mark.integration
NOW = datetime(2026, 9, 10, tzinfo=UTC)


class Provider:
    version = "test-events-v1"
    dimensions = 2

    def embed(self, _):
        return [1, 0]


def insert(sessions, id_, title, *, hours=0, published=True):
    with sessions.begin() as session:
        session.add(
            ArticleRecord(
                id=id_,
                title=title,
                url=f"https://example.com/events/{id_}",
                description="A news article reporting the announcement.",
                source=f"Publisher {id_}",
                category="Technology",
                language="en",
                retrieved_at=NOW + timedelta(hours=hours),
                published_at=NOW + timedelta(hours=hours) if published else None,
                cleaning_version="1",
                processing_status="clean",
            )
        )


@pytest.fixture
def setup(database):
    sessions = create_session_factory(database[0])
    provider = Provider()
    client = QdrantClient(":memory:")
    store = QdrantEmbeddingStore(client, "event_test", provider.dimensions)
    try:
        insert(sessions, 1, "NVIDIA launches a new GPU")
        insert(sessions, 2, "NVIDIA unveils a new GPU", hours=2)
        insert(sessions, 3, "NVIDIA reports quarterly earnings", hours=3)
        embeddings = ArticleEmbeddingService(sessions, provider, store)
        assert embeddings.sync()["indexed"] == 3
        yield (
            sessions,
            provider,
            store,
            embeddings,
            EventClusteringService(sessions, provider, store),
        )
    finally:
        client.close()


def test_same_event_growth_idempotence_provenance_and_http(setup):
    sessions, _, _, embeddings, clustering = setup
    first = clustering.sync()
    assert first["events"] == 2 and first["multi_article_events"] == 1
    reader = EventReadService(sessions)
    event = reader.list(EventListRequest()).results[0]
    assert event.article_count == event.source_count == 2
    assert event.summary is None and not event.stale
    assert event.articles[0].evidence["weakest_pair"]["compatible"]
    repeated = clustering.sync()
    assert repeated["events_changed"] == 0 and repeated["assignments"] == first["assignments"]
    assert reader.get(event.id).last_updated_at == event.last_updated_at
    insert(sessions, 4, "NVIDIA reveals a new GPU", hours=4)
    embeddings.sync()
    third = clustering.sync()
    assert third["assignments"]["4"] == event.id and third["events"] == 2
    detail = reader.get(event.id)
    assert detail.article_count == 3 and detail.first_published_at == NOW
    assert detail.last_published_at == NOW + timedelta(hours=4)
    assert detail.last_updated_at > event.last_updated_at
    with TestClient(create_app(events=reader.list, event_detail=reader.get)) as api:
        assert api.get("/events").json()["count"] == 1
        assert api.get("/events?min_articles=1").json()["total"] == 2
        assert api.get(f"/events/{event.id}").json()["article_count"] == 3
    assert len(reader.runs()) == 3


def test_stale_article_flagged_then_detached_and_reindexed(setup):
    sessions, _, _, embeddings, clustering = setup
    first = clustering.sync()
    reader = EventReadService(sessions)
    event_id = first["assignments"]["1"]
    with sessions.begin() as session:
        session.get(ArticleRecord, 1).title = "NVIDIA reports quarterly earnings"
    assert reader.get(event_id).stale
    result = clustering.sync()
    assert result["excluded"] == {"1": "stale_or_missing_ledger"}
    assert "1" not in result["assignments"]
    embeddings.sync()
    result = clustering.sync()
    assert not result["excluded"]
    assert result["assignments"]["1"] == result["assignments"]["3"]
    assert result["assignments"]["1"] != result["assignments"]["2"]


def test_missing_vector_or_unknown_date_never_silently_join(setup):
    sessions, _, store, embeddings, clustering = setup
    store.delete(2)
    insert(sessions, 4, "NVIDIA launches a new GPU", published=False)
    embeddings.sync()
    store.delete(2)
    result = clustering.sync()
    assert result["excluded"] == {"2": "missing_vector"}
    assert result["events"] == 3
    assert clustering.compare_articles(1, 2)["reason"] == "missing_vector"
    assert "missing_publication_date" in clustering.compare_articles(1, 4)["reasons"]


def test_changed_publication_dates_split_events_even_with_unchanged_text(setup):
    sessions, _, _, _, clustering = setup
    first = clustering.sync()
    with sessions.begin() as session:
        session.get(ArticleRecord, 2).published_at = NOW + timedelta(days=30)
    assert EventReadService(sessions).get(first["assignments"]["1"]).stale
    result = clustering.sync()
    assert result["events"] == 3 and result["assignments"]["1"] != result["assignments"]["2"]


def test_corpus_limit_and_backend_failure_preserve_committed_events(setup, monkeypatch):
    sessions, provider, store, _, clustering = setup
    before = clustering.sync()
    with pytest.raises(ValueError):
        EventClusteringService(sessions, provider, store, ClusterConfig(max_articles=2)).sync()
    monkeypatch.setattr(store, "get", lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(RuntimeError):
        clustering.sync()
    assert EventReadService(sessions).runs()[0]["report"] == before
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(EventClusterRun)) == 1
        assert session.scalar(select(func.count()).select_from(ArticleEventRecord)) == 3


def test_membership_replacement_rolls_back_atomically(setup, monkeypatch):
    sessions, _, _, _, clustering = setup
    original = clustering.sync()
    import backend.app.services.event_clustering as module

    monkeypatch.setattr(module, "compare", lambda *_: (_ for _ in ()).throw(RuntimeError("fail")))
    with pytest.raises(RuntimeError):
        clustering.sync()
    with sessions() as session:
        assert {
            str(link.article_id): link.event_id
            for link in session.scalars(select(ArticleEventRecord))
        } == original["assignments"]
        assert session.scalar(select(func.count()).select_from(EventClusterRun)) == 1


def test_deletion_cascades_and_empty_events_retire(setup):
    sessions, _, _, _, clustering = setup
    result = clustering.sync()
    event_id = result["assignments"]["1"]
    with sessions.begin() as session:
        session.delete(session.get(ArticleRecord, 1))
    assert EventReadService(sessions).get(event_id).stale
    clustering.sync()
    assert not EventReadService(sessions).get(event_id).stale
    with sessions.begin() as session:
        session.delete(session.get(ArticleRecord, 2))
    result = clustering.sync()
    assert event_id in result["retired_event_ids"]
    assert EventReadService(sessions).get(event_id) is None


def test_migration_round_trip_preserves_articles(database):
    engine, config, _ = database
    sessions = create_session_factory(engine)
    insert(sessions, 1, "A test article for event migration")
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.check(config)
        command.downgrade(config, "0011")
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 1
        command.upgrade(config, "head")
        command.check(config)
        assert connection.scalar(text("SELECT count(*) FROM events")) == 0


def test_ingestion_event_failure_does_not_undo_articles(database, feed, rss_bytes, caplog):
    sessions = create_session_factory(database[0])
    order = []

    class Fetcher:
        def fetch(self, _):
            return rss_bytes

    def embeddings():
        order.append("embeddings")
        return {"indexed": 3}

    def events():
        order.append("events")
        raise RuntimeError("private credentials")

    outcome = NewsUpdateService(
        sessions,
        [feed],
        RSSCollector(Fetcher()).collect,
        sync_embeddings=embeddings,
        sync_events=events,
    ).run()
    assert outcome.succeeded and outcome.report.inserted == 3
    assert order == ["embeddings", "events"]
    assert "news_update_events_deferred" in caplog.text and "private credentials" not in caplog.text
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(ArticleRecord)) == 3
        assert session.scalar(select(func.count()).select_from(EventRecord)) == 0


def test_event_entities_deduplicate_with_member_provenance_and_exclude_stale(setup):
    from backend.app.models.entity import EntityRecord, article_entities

    sessions, _, _, _, clustering = setup
    with sessions.begin() as session:
        shared = EntityRecord(name="NVIDIA", canonical_name="nvidia", entity_type="ORGANIZATION")
        unrelated = EntityRecord(name="Other", canonical_name="other", entity_type="PERSON")
        session.add_all([shared, unrelated])
        session.flush()
        shared_id = shared.id
        session.execute(
            article_entities.insert(),
            [
                {"article_id": 1, "entity_id": shared.id},
                {"article_id": 2, "entity_id": shared.id},
                {"article_id": 3, "entity_id": unrelated.id},
            ],
        )
    event_id = clustering.sync()["assignments"]["1"]
    reader = EventReadService(sessions)
    detail = reader.get(event_id)
    assert len(detail.related_entities) == 1
    entity = detail.related_entities[0]
    assert entity.id == shared_id and entity.name == "NVIDIA"
    assert entity.article_ids == [1, 2]
    with TestClient(create_app(event_detail=reader.get)) as api:
        assert api.get(f"/events/{event_id}").json()["related_entities"][0]["article_ids"] == [1, 2]
    with sessions.begin() as session:
        session.get(ArticleRecord, 1).processing_status = "rejected"
    assert reader.get(event_id).related_entities[0].article_ids == [2]
    with sessions.begin() as session:
        session.get(ArticleRecord, 2).title = "Changed after grouping"
    assert reader.get(event_id).related_entities == []
