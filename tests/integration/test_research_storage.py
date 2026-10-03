from datetime import UTC, date, datetime

import pytest
from sqlalchemy import update

from backend.app.database.session import create_session_factory
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchRequest
from backend.app.services.chat_context import record_evidence
from backend.app.services.research_tools import ResearchTools, article_fingerprint

pytestmark = pytest.mark.integration
NOW = datetime(2026, 9, 18, tzinfo=UTC)


@pytest.fixture
def stored(database):
    engine, _, settings = database
    sessions = create_session_factory(engine)
    with sessions.begin() as session:
        record = ArticleRecord(
            title="Orion shares fell after profit warning",
            description="Orion cut its profit forecast.",
            content="The company cited lower demand for its solar panels.",
            url="https://example.com/orion",
            source="Daily",
            retrieved_at=NOW,
            published_at=NOW,
            category="business",
            language="en",
            cleaning_version="test-v1",
            processing_status="cleaned",
        )
        session.add(record)
        session.flush()
        id_ = record.id
        evidence, fingerprint = record_evidence(record), article_fingerprint(record)
    scope = SearchRequest(
        q="Orion",
        recency="off",
        source="Daily",
        language="en",
        category="business",
        date_from=date(2026, 9, 18),
        date_to=date(2026, 9, 18),
    )
    tools = ResearchTools(settings, scope, NOW)
    tools.known[id_] = evidence
    tools.fingerprints[id_] = fingerprint
    return sessions, tools, id_, evidence


def test_real_article_read_is_scoped_and_unchanged(stored):
    _, tools, id_, evidence = stored
    assert tools.get_article(id_) == evidence
    with pytest.raises(ValueError, match="observed"):
        tools.get_article(id_ + 100)
    with pytest.raises(ValueError, match="widen"):
        tools.search_news("Orion", "Wire")


@pytest.mark.parametrize(
    "change",
    [
        {"source": "Wire"},
        {"language": "ms"},
        {"category": "technology"},
        {"published_at": datetime(2026, 9, 17, tzinfo=UTC)},
        {"processing_status": "rejected"},
        {"cleaning_version": None},
        {"content": "Changed text must not use the old ranking."},
    ],
)
def test_changed_and_out_of_scope_articles_are_rejected(stored, change):
    sessions, tools, id_, _ = stored
    with sessions.begin() as session:
        session.execute(update(ArticleRecord).where(ArticleRecord.id == id_).values(**change))
    assert tools.get_article(id_) == []


def test_deleted_article_cannot_be_cited(stored):
    sessions, tools, id_, _ = stored
    with sessions.begin() as session:
        session.delete(session.get(ArticleRecord, id_))
    assert tools.get_article(id_) == []
