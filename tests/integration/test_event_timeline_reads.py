from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.database.session import create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.models.event import ArticleEventRecord, EventRecord
from backend.app.schemas.event_timeline import TimelineRequest
from backend.app.services.event_clustering import event_input_hash
from backend.app.services.events import EventReadService

pytestmark = pytest.mark.integration
NOW = datetime(2026, 9, 12, 9, tzinfo=UTC)


def article(id_, published_at):
    return ArticleRecord(
        id=id_,
        title="Orion announced the acquisition on Saturday.",
        description="Orion announced the acquisition on Saturday. Regulators will review the deal.",
        source="City Wire",
        url=f"https://example.com/timeline/{id_}",
        category="Business",
        language="en",
        published_at=published_at,
        retrieved_at=NOW,
        cleaning_version="1",
        processing_status="clean",
    )


@pytest.fixture
def timeline_setup(database):
    sessions = create_session_factory(database[0])
    with sessions.begin() as session:
        articles = [article(1, NOW), article(2, NOW + timedelta(hours=2)), article(3, None)]
        session.add_all(articles)
        session.flush()
        session.add(
            EventRecord(
                id=1,
                title=articles[0].title,
                representative_article_id=1,
                first_seen_at=NOW,
                first_published_at=NOW,
                last_published_at=NOW + timedelta(hours=2),
                last_updated_at=NOW,
                policy_version="test-cluster",
                model_version="test-model",
            )
        )
        session.flush()
        session.add_all(
            [
                ArticleEventRecord(
                    article_id=row.id,
                    event_id=1,
                    similarity_score=0.9,
                    input_hash=event_input_hash(row),
                    evidence={"compared_members": 2},
                )
                for row in articles
            ]
        )
    return sessions, EventReadService(sessions)


def test_read_only_timeline_http_paging_and_no_summary_cache(timeline_setup):
    sessions, reader = timeline_setup
    first = reader.timeline(1, TimelineRequest(limit=2))
    second = reader.timeline(1, TimelineRequest(limit=2, offset=2))
    assert first.status == "ready" and first.document.next_offset == 2
    assert [item.source_article_id for item in first.document.items] == [1, 2]
    assert second.document.undated_items[0].source_article_id == 3
    assert second.input_hash == first.input_hash
    assert reader.timeline(99, TimelineRequest()) is None
    with sessions() as session:
        event = session.get(EventRecord, 1)
        assert event.summary is None and event.last_updated_at == NOW
    with TestClient(create_app(event_timeline=reader.timeline)) as client:
        response = client.get("/events/1/timeline?limit=2&offset=2")
        assert response.status_code == 200
        assert response.json()["document"] == second.document.model_dump(mode="json")


@pytest.mark.parametrize("change", ["description", "published_at", "delete", "rejected"])
def test_changed_members_suppress_timeline(timeline_setup, change):
    sessions, reader = timeline_setup
    with sessions.begin() as session:
        row = session.get(ArticleRecord, 2)
        if change == "delete":
            session.delete(row)
        elif change == "rejected":
            row.processing_status = "rejected"
        elif change == "published_at":
            row.published_at = NOW - timedelta(days=1)
        else:
            row.description = "The regulator rejected the acquisition on Saturday."
    view = reader.timeline(1, TimelineRequest())
    assert view.status == "stale" and view.stale and view.document is None


def test_new_report_and_reconciled_edits_update_current_timeline(timeline_setup):
    sessions, reader = timeline_setup
    first = reader.timeline(1, TimelineRequest())
    with sessions.begin() as session:
        row = session.get(ArticleRecord, 2)
        row.url = "https://example.com/updated-report"
        row.source = "Updated Publisher"
        row.description = "The regulator issued a statement about its review."
        session.get(ArticleEventRecord, 2).input_hash = event_input_hash(row)
        new = article(4, NOW + timedelta(hours=4))
        session.add(new)
        session.flush()
        for link in session.scalars(select(ArticleEventRecord)):
            link.evidence = {"compared_members": 3}
        session.add(
            ArticleEventRecord(
                article_id=4,
                event_id=1,
                similarity_score=0.9,
                input_hash=event_input_hash(new),
                evidence={"compared_members": 3},
            )
        )
    second = reader.timeline(1, TimelineRequest())
    assert second.status == "ready" and second.document.total_count == 4
    assert second.input_hash != first.input_hash
    assert second.document.items[1].id == first.document.items[1].id
    assert str(second.document.items[1].citation.url).endswith("updated-report")
    assert "regulator issued" in second.document.items[1].description


def test_no_usable_text_is_distinguished_from_empty_page(timeline_setup):
    sessions, reader = timeline_setup
    with sessions.begin() as session:
        for row in session.scalars(select(ArticleRecord)):
            row.language = "ms"
            session.get(ArticleEventRecord, row.id).input_hash = event_input_hash(row)
    view = reader.timeline(1, TimelineRequest(offset=99))
    assert view.status == "insufficient_evidence"
    assert view.document.total_count == 0 and len(view.document.excluded_articles) == 3
