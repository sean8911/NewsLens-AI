from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.database.session import create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.models.event import ArticleEventRecord, EventRecord
from backend.app.services.event_clustering import event_input_hash
from backend.app.services.events import EventReadService

pytestmark = pytest.mark.integration
NOW = datetime(2026, 9, 12, tzinfo=UTC)


@pytest.fixture
def comparison_setup(database):
    sessions = create_session_factory(database[0])
    with sessions.begin() as session:
        articles = [
            ArticleRecord(
                id=id_,
                title="Orion announced the acquisition on Thursday.",
                description=(
                    "Orion announced the acquisition on Thursday. "
                    "The deal requires regulatory approval."
                ),
                source=f"Publisher {id_}",
                url=f"https://example.com/{id_}",
                category="Business",
                language="en",
                published_at=NOW,
                retrieved_at=NOW,
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
                policy_version="test-cluster",
                model_version="test-model",
            )
        )
        session.flush()
        session.add_all(
            [
                ArticleEventRecord(
                    article_id=article.id,
                    event_id=1,
                    similarity_score=0.9,
                    input_hash=event_input_hash(article),
                    evidence={"compared_members": 1},
                )
                for article in articles
            ]
        )
    return sessions, EventReadService(sessions)


def test_current_comparison_without_summary_cache_and_no_writes(comparison_setup):
    sessions, reader = comparison_setup
    first, second = reader.comparison(1), reader.comparison(1)
    assert first.status == "compared" and first.document.agreement
    assert first.input_hash == second.input_hash and first.document == second.document
    assert reader.comparison(99) is None
    with sessions() as session:
        assert session.get(EventRecord, 1).last_updated_at == NOW
        assert session.get(EventRecord, 1).summary is None
    with TestClient(create_app(event_comparison=reader.comparison)) as client:
        assert client.get("/events/1/comparison").json()["document"] == first.document.model_dump(
            mode="json"
        )
        assert client.get("/events/99/comparison").status_code == 404


@pytest.mark.parametrize("change", ["description", "delete", "rejected"])
def test_changed_or_missing_members_hide_comparison(comparison_setup, change):
    sessions, reader = comparison_setup
    with sessions.begin() as session:
        article = session.get(ArticleRecord, 2)
        if change == "delete":
            session.delete(article)
        elif change == "description":
            article.description = "The company cancelled the acquisition on Friday."
        else:
            article.processing_status = "rejected"
    view = reader.comparison(1)
    assert view.status == "stale" and view.stale and view.document is None


def test_metadata_edits_and_reconciled_content_take_effect_immediately(comparison_setup):
    sessions, reader = comparison_setup
    first = reader.comparison(1)
    with sessions.begin() as session:
        session.get(ArticleRecord, 2).source = "Publisher 1"
    assert reader.comparison(1).status == "insufficient_sources"
    with sessions.begin() as session:
        article = session.get(ArticleRecord, 2)
        article.source = "Updated Publisher"
        article.url = "https://example.com/updated"
        article.description = (
            "Orion announced the acquisition on Thursday. Employees requested job guarantees."
        )
        # Simulate the membership fingerprint written by a successful reconciliation.
        session.get(ArticleEventRecord, 2).input_hash = event_input_hash(article)
    result = reader.comparison(1)
    assert result.status == "compared" and result.input_hash != first.input_hash
    assert any("Employees" in fact.text for fact in result.document.additional_details)
    assert any(str(cite.url).endswith("/updated") for cite in result.document.citations)
    with sessions() as session:
        assert len(list(session.scalars(select(ArticleRecord)))) == 2
