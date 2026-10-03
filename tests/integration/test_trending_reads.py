from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.database.session import create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.models.article_topic import ArticleTopicRecord
from backend.app.models.entity import EntityRecord, article_entities
from backend.app.schemas.trending import TrendingRequest
from backend.app.services.trending import TrendingCapacityError, TrendingService
from nlp.classification.classifier import TopicClassifier
from nlp.entities.extractor import EntityExtractor

pytestmark = pytest.mark.integration
NOW = datetime(2026, 9, 12, 12, tzinfo=UTC)


@pytest.fixture
def trend_setup(database):
    sessions = create_session_factory(database[0])
    classifier, extractor = TopicClassifier(), EntityExtractor()
    with sessions.begin() as session:
        entity = EntityRecord(name="NVIDIA", canonical_name="nvidia", entity_type="ORGANIZATION")
        session.add(entity)
        for id_, hours in enumerate((1, 2, 24, 25, 48, 49, 0, -1, None), 1):
            row = ArticleRecord(
                id=id_,
                title="NVIDIA unveils a new semiconductor chip",
                description="NVIDIA introduced a processor for artificial intelligence systems.",
                category="Technology",
                language="en",
                source=f"Publisher {id_}",
                url=f"https://example.com/trend/{id_}",
                retrieved_at=NOW,
                published_at=NOW - timedelta(hours=hours) if hours is not None else None,
                cleaning_version="1",
                processing_status="clean",
                topic_status="classified",
                entity_status="extracted",
            )
            row.topic_input_hash = classifier.prepare(row.title, row.description, None, "en")[2]
            row.entity_input_hash = extractor.prepare(row.title, row.description, None, "en")[2]
            row.article_topics = [ArticleTopicRecord(topic="Semiconductors", confidence=0.9)]
            session.add(row)
            session.flush()
            session.execute(
                article_entities.insert().values(article_id=row.id, entity_id=entity.id)
            )
    return sessions, TrendingService(sessions)


def test_sql_boundaries_counts_http_and_read_only(trend_setup):
    sessions, service = trend_setup
    with sessions() as session:
        before = list(
            session.scalars(text("SELECT row_to_json(a)::text FROM articles a ORDER BY id"))
        )
    request = TrendingRequest(as_of=NOW)
    result = service.read(request)
    assert result.current_articles == 3 and result.previous_articles == 2
    assert result.articles_scanned == 5
    assert (
        result.results[0].growth_percent == 50 and result.results[0].entities[0].article_count == 3
    )
    with TestClient(create_app(trending=service.read)) as client:
        response = client.get("/trending", params={"as_of": NOW.isoformat()})
        assert response.status_code == 200 and response.json() == result.model_dump(mode="json")
    with sessions() as session:
        assert (
            list(session.scalars(text("SELECT row_to_json(a)::text FROM articles a ORDER BY id")))
            == before
        )


@pytest.mark.parametrize("change", ["text", "topic_hash", "rejected"])
def test_stale_or_rejected_articles_cannot_inflate_trends(trend_setup, change):
    sessions, service = trend_setup
    with sessions.begin() as session:
        row = session.get(ArticleRecord, 1)
        if change == "text":
            row.description = "The company abandoned the processor development project."
        elif change == "topic_hash":
            row.topic_input_hash = "outdated"
        else:
            row.processing_status = "rejected"
    result = service.read(TrendingRequest(as_of=NOW))
    assert result.current_articles == 2 and sum(result.excluded.values()) == 1
    assert all(e.article_id != 1 for e in result.results[0].evidence)


def test_entity_freshness_and_publisher_edits_affect_scores(trend_setup):
    sessions, service = trend_setup
    first = service.read(TrendingRequest(as_of=NOW))
    with sessions.begin() as session:
        session.get(ArticleRecord, 1).entity_input_hash = "outdated"
    result = service.read(TrendingRequest(as_of=NOW))
    assert (
        result.results[0].components["entity_frequency"]
        < first.results[0].components["entity_frequency"]
    )
    assert result.input_hash != first.input_hash
    with sessions.begin() as session:
        for id_ in (1, 2, 3):
            session.get(ArticleRecord, id_).source = "ONE Publisher"
    assert service.read(TrendingRequest(as_of=NOW)).status == "no_trends"


def test_complete_window_capacity_is_checked_before_scoring(trend_setup):
    sessions, _ = trend_setup
    with pytest.raises(TrendingCapacityError):
        TrendingService(sessions, max_articles=4).read(TrendingRequest(as_of=NOW))


def test_empty_window_does_not_relabel_historical_articles_as_current(trend_setup):
    _, service = trend_setup
    result = service.read(TrendingRequest(as_of=NOW + timedelta(days=20)))
    assert result.status == "no_trends" and not result.results
    assert result.current_articles == result.previous_articles == result.articles_scanned == 0
