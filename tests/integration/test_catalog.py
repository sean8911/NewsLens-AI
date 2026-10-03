from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from backend.app.catalog_runtime import CatalogRuntime, read_system_status
from backend.app.database.session import create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.models.article_topic import ArticleTopicRecord
from backend.app.models.entity import EntityRecord, article_entities

pytestmark = pytest.mark.integration


@pytest.fixture
def catalog(database):
    engine, _, settings = database
    ids = []
    with create_session_factory(engine).begin() as session:
        entity = EntityRecord(name="ACME_100%", canonical_name="acme_100%", entity_type="PRODUCT")
        hidden = EntityRecord(name="Hidden", canonical_name="hidden", entity_type="PERSON")
        session.add_all([entity, hidden])
        session.flush()
        for i in range(5):
            article = ArticleRecord(
                title=f"Article {i}",
                description="A stored description.",
                source="Publisher" if i != 1 else "Other",
                category="Technology",
                language="en" if i != 2 else None,
                url=f"https://example.com/catalog/{i}",
                retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
                published_at=datetime(2026, 9, 19, tzinfo=UTC) if i != 2 else None,
                cleaning_version="1" if i != 4 else None,
                processing_status="clean" if i < 3 else "rejected" if i == 3 else "legacy",
            )
            session.add(article)
            session.flush()
            ids.append(article.id)
            session.add(ArticleTopicRecord(article_id=article.id, topic="AI", confidence=0.9))
            session.execute(
                article_entities.insert().values(
                    article_id=article.id, entity_id=entity.id if i < 3 else hidden.id
                )
            )
        entity_id, hidden_id = entity.id, hidden.id
    with TestClient(
        create_app(
            catalog=CatalogRuntime(settings), system_status=lambda: read_system_status(settings)
        )
    ) as client:
        yield client, ids, entity_id, hidden_id


def test_article_pages_have_consistent_totals_order_and_visibility(catalog):
    client, ids, _, _ = catalog
    first = client.get("/articles?limit=1").json()
    assert first["count"] == 1 and first["total"] == 3 and first["has_more"]
    assert first["results"][0]["id"] == ids[1]  # Equal dates: stable descending ID.
    second = client.get("/articles?limit=1&offset=1").json()
    assert second["results"][0]["id"] == ids[0]
    last = client.get("/articles?limit=1&offset=2").json()
    assert last["results"][0]["id"] == ids[2] and not last["has_more"]
    beyond = client.get("/articles?offset=99").json()
    assert beyond["results"] == [] and beyond["total"] == 3 and not beyond["has_more"]
    assert "content" not in first["results"][0]
    for article_id in ids[3:]:
        assert client.get(f"/articles/{article_id}").status_code == 404
    detail = client.get(f"/articles/{ids[0]}").json()
    assert "content" in detail and "keywords" in detail and "topic_input_hash" not in detail


def test_article_filters_dates_and_entity_scope(catalog):
    client, ids, entity_id, _ = catalog
    query = {
        "source": "Publisher",
        "category": "Technology",
        "language": "en",
        "topic": "AI",
        "entity_id": entity_id,
        "date_from": "2026-09-19",
        "date_to": "2026-09-19",
    }
    response = client.get("/articles", params=query)
    assert response.status_code == 200
    assert [x["id"] for x in response.json()["results"]] == [ids[0]]
    assert client.get("/articles?language=unknown").json()["results"][0]["id"] == ids[2]
    assert client.get("/articles?topic=Missing").json()["total"] == 0
    assert client.get("/articles?date_from=2026-09-20").json()["total"] == 0


def test_topics_entities_counts_and_literal_names(catalog):
    client, ids, entity_id, hidden_id = catalog
    assert client.get("/topics").json()["results"] == [{"topic": "AI", "article_count": 3}]
    assert client.get("/topics?offset=1").json()["results"] == []
    entities = client.get("/entities?name=100%25&entity_type=PRODUCT").json()
    assert entities["total"] == 1 and entities["results"][0]["article_count"] == 3
    assert client.get("/entities?name=100_").json()["total"] == 0
    assert client.get(f"/entities/{hidden_id}").status_code == 404
    detail = client.get(f"/entities/{entity_id}?limit=1&offset=2").json()
    assert detail["entity"]["article_count"] == detail["articles"]["total"] == 3
    assert detail["articles"]["results"][0]["id"] == ids[2]
    assert not detail["articles"]["has_more"]


def test_status_probes_real_database_without_loading_models(catalog):
    client, _, _, _ = catalog
    response = client.get("/system/status")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] == "available" and body["article_count"] == 3
    assert body["last_successful_news_update"] is None
    assert body["model_runtime"] == body["vector_runtime"] == "not_checked"
    assert body["latest_article_published_at"] == "2026-09-19T00:00:00Z"
    assert body["latest_article_retrieved_at"] == "2026-09-20T00:00:00Z"


def test_freshness_excludes_hidden_articles_and_preserves_partial_success(database, catalog):
    from backend.app.models.ingestion_run import IngestionRun

    engine, _, settings = database
    _, ids, _, _ = catalog
    with create_session_factory(engine).begin() as session:
        hidden = session.get(ArticleRecord, ids[3])
        hidden.published_at = hidden.retrieved_at = datetime(2099, 1, 1, tzinfo=UTC)
        for hour, status in [(1, "succeeded"), (2, "partial"), (3, "failed"), (4, "running")]:
            session.add(
                IngestionRun(
                    started_at=datetime(2026, 9, 20, hour, tzinfo=UTC),
                    finished_at=None
                    if status == "running"
                    else datetime(2026, 9, 20, hour, 5, tzinfo=UTC),
                    status=status,
                    feeds_checked=2,
                )
            )
    result = read_system_status(settings)
    assert result.last_successful_news_update == datetime(2026, 9, 20, 2, 5, tzinfo=UTC)
    assert result.last_successful_update_status == "partial"
    assert result.last_update_attempt_status == "running"
    assert result.last_update_attempt_at == datetime(2026, 9, 20, 4, tzinfo=UTC)
    assert result.latest_article_published_at == datetime(2026, 9, 19, tzinfo=UTC)
    assert result.latest_article_retrieved_at == datetime(2026, 9, 20, tzinfo=UTC)


def test_empty_library_does_not_invent_a_freshness_timestamp(database):
    _, _, settings = database
    result = read_system_status(settings)
    assert result.status == "ok" and result.article_count == 0
    assert result.latest_article_published_at is None
    assert result.latest_article_retrieved_at is None
    assert result.last_successful_news_update is None
    assert result.last_update_attempt_status is None
