from datetime import UTC, datetime

import pytest
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from backend.app.database.session import create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.models.event import ArticleEventRecord, EventRecord
from backend.app.models.event_summary import EventSummaryRecord
from backend.app.services.event_clustering import event_input_hash
from backend.app.services.event_summaries import (
    EventSummaryService,
    prepare_summary,
    validate_grounding,
)
from backend.app.services.events import EventReadService
from nlp.events.summarization import EventSummarizer

pytestmark = pytest.mark.integration
NOW = datetime(2026, 9, 10, tzinfo=UTC)


def populate(sessions):
    with sessions.begin() as session:
        articles = [
            ArticleRecord(
                id=id_,
                title="Orion announces a new AI processor",
                description=(
                    "Orion introduced an AI processor in Penang. "
                    + (
                        "The company plans to hire 200 engineers."
                        if id_ == 1
                        else "Local authorities will review the factory building plans."
                    )
                ),
                url=f"https://example.com/{id_}",
                source=f"Publisher {id_}",
                published_at=NOW,
                retrieved_at=NOW,
                language="en",
                category="Technology",
                cleaning_version="1",
                processing_status="clean",
            )
            for id_ in (1, 2)
        ]
        session.add_all(articles)
        session.flush()
        session.add(
            EventRecord(
                id=1,
                title=articles[0].title,
                representative_article_id=1,
                first_seen_at=NOW,
                first_published_at=NOW,
                last_published_at=NOW,
                last_updated_at=NOW,
                policy_version="cluster-v1",
                model_version="model-v1",
            )
        )
        session.flush()
        for article in articles:
            session.add(
                ArticleEventRecord(
                    article_id=article.id,
                    event_id=1,
                    similarity_score=0.9,
                    input_hash=event_input_hash(article),
                    evidence={"compared_members": 1},
                )
            )


@pytest.fixture
def setup(database):
    sessions = create_session_factory(database[0])
    populate(sessions)
    return sessions, EventSummaryService(sessions), EventReadService(sessions)


def test_persistence_read_api_and_repeat_idempotence(setup):
    sessions, service, reader = setup
    assert service.refresh(1)["status"] == "summarized"
    event = reader.get(1)
    assert event.summary and event.event_summary.document.source_count == 2
    assert not service.refresh(1)["changed"]
    assert reader.get(1).event_summary.generated_at == event.event_summary.generated_at
    assert reader.get(1).last_updated_at == NOW  # Clustering timestamp is independent.
    with TestClient(create_app(events=reader.list, event_detail=reader.get)) as api:
        result = api.get("/events/1").json()
        assert result["event_summary"]["status"] == "summarized"
        assert api.get("/events").json()["results"][0]["summary"] == result["summary"]
    with sessions() as session:
        rows = session.execute(
            select(ArticleEventRecord, ArticleRecord).join(
                ArticleRecord, ArticleRecord.id == ArticleEventRecord.article_id
            )
        ).all()
        sources, _ = prepare_summary(session.get(EventRecord, 1), rows)
        validate_grounding(event.event_summary.document, sources)


@pytest.mark.parametrize("change", ["content", "source", "url", "delete"])
def test_source_changes_hide_old_summary_before_refresh(setup, change):
    sessions, service, reader = setup
    service.refresh(1)
    with sessions.begin() as session:
        article = session.get(ArticleRecord, 2)
        if change == "delete":
            session.delete(article)
        else:
            setattr(
                article,
                change,
                {
                    "content": "Different information now appears here.",
                    "source": "Publisher 1",
                    "url": "https://example.com/changed",
                }[change],
            )
    result = reader.get(1)
    assert result.summary is None and result.event_summary.status == "stale"
    assert result.event_summary.document is None
    status = service.refresh(1)["status"]
    assert (
        status
        == {
            "content": "stale",
            "source": "insufficient_sources",
            "delete": "stale",
            "url": "summarized",
        }[change]
    )


def test_membership_growth_invalidates_summary_and_refresh_includes_new_source(setup):
    sessions, service, reader = setup
    service.refresh(1)
    with sessions.begin() as session:
        article = ArticleRecord(
            id=3,
            title="Orion announces a new AI processor",
            description="Orion said construction would begin next year.",
            url="https://example.com/3",
            source="Publisher 3",
            published_at=NOW,
            retrieved_at=NOW,
            category="Technology",
            language="en",
            cleaning_version="1",
            processing_status="clean",
        )
        session.add(article)
        session.flush()
        for link in session.scalars(select(ArticleEventRecord)):
            link.evidence = {"compared_members": 2}
        session.add(
            ArticleEventRecord(
                article_id=3,
                event_id=1,
                similarity_score=0.9,
                input_hash=event_input_hash(article),
                evidence={"compared_members": 2},
            )
        )
    assert reader.get(1).event_summary.stale
    assert service.refresh(1)["changed"]
    assert reader.get(1).event_summary.document.source_count == 3


def test_failure_does_not_delete_articles_or_events_and_can_retry(setup, caplog):
    sessions, service, reader = setup
    service.refresh(1)

    class Broken(EventSummarizer):
        def summarize(self, _):
            raise RuntimeError("private database credentials")

    assert EventSummaryService(sessions, Broken()).refresh(1, force=True)["status"] == "error"
    assert reader.get(1).summary is None
    assert "private database credentials" not in caplog.text
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(ArticleRecord)) == 2
        assert session.scalar(select(func.count()).select_from(EventRecord)) == 1
    assert service.refresh(1)["status"] == "summarized"


def test_fabricated_summary_is_rejected_at_storage_boundary(setup):
    sessions, _, reader = setup

    class Ungrounded(EventSummarizer):
        def summarize(self, sources):
            status, document = super().summarize(sources)
            document.what_happened.text = "The company earned a billion dollars."
            return status, document

    result = EventSummaryService(sessions, Ungrounded()).refresh(1)
    assert result["status"] == "error" and reader.get(1).summary is None


def test_event_deletion_cascades_summary_and_missing_event_is_explicit(setup):
    sessions, service, _ = setup
    service.refresh(1)
    with sessions.begin() as session:
        session.delete(session.get(EventRecord, 1))
    assert service.refresh(1)["status"] == "not_found"
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(EventSummaryRecord)) == 0


def test_migration_roundtrip_preserves_events_and_articles(database):
    engine, config, _ = database
    sessions = create_session_factory(engine)
    populate(sessions)
    EventSummaryService(sessions).refresh(1)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.check(config)
        command.downgrade(config, "0012")
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 2
        assert connection.scalar(text("SELECT summary FROM events WHERE id=1")) is None
        command.upgrade(config, "head")
        command.check(config)
