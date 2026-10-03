from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from backend.app.database.session import create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchRequest
from backend.app.search_runtime import search_snapshot

pytestmark = pytest.mark.integration
NOW = datetime(2026, 9, 11, 12, tzinfo=UTC)


@pytest.fixture
def dated_articles(database, monkeypatch):
    settings = database[2]
    sessions = create_session_factory(database[0])
    ids = []
    with sessions.begin() as session:
        for index in range(33):
            article = ArticleRecord(
                title=f"NVIDIA shipment report {index}",
                content="Fresh delivery report." if index == 29 else "Old delivery report.",
                source="Example" if index < 30 else "Excluded",
                language="en",
                category="Technology",
                published_at=datetime(2026, 9, 10, tzinfo=UTC)
                if index == 29
                else datetime(2026, 1, 10, tzinfo=UTC),
                retrieved_at=NOW,
                cleaning_version="1",
                processing_status="clean",
                url=f"https://example.com/dated/{index}",
            )
            session.add(article)
            session.flush()
            ids.append(article.id)

    class Reranker:
        version = "dated-reranker"

        def score(self, query, documents):
            return [5.0 if "Fresh" in document else 6.0 for document in documents]

    monkeypatch.setattr("backend.app.search_runtime.reranker_for", lambda _: Reranker())
    return settings, ids


def search(request, settings):
    with search_snapshot(request, settings, as_of=NOW) as (response, _):
        return response


def test_recency_runs_before_final_limit_and_keeps_filters(dated_articles):
    settings, ids = dated_articles
    request = SearchRequest(
        q="NVIDIA shipment",
        method="lexical",
        rerank=True,
        limit=5,
        recency="latest",
        source="Example",
        language="en",
        category="Technology",
    )
    off = search(request.model_copy(update={"recency": "off"}), settings)
    recent = search(request, settings)
    assert ids[29] not in [hit.article_id for hit in off.results]
    assert recent.results[0].article_id == ids[29]
    assert recent.reranking.candidate_count == len(recent.temporal.scores) == 30
    assert all(hit.source == "Example" for hit in recent.results)
    assert recent.reranking.scores[-1].reranking_score == 5.0
    assert recent.results[0].relevance_score != 5.0


def test_frozen_research_intent_keeps_recency_across_followup_queries(dated_articles):
    from backend.app.services.time_intent import resolve_time_intent

    settings, ids = dated_articles
    original = SearchRequest(q="Latest NVIDIA shipment", recency="latest", method="lexical")
    intent = resolve_time_intent(original, NOW)
    followup = SearchRequest(
        q="NVIDIA shipment", recency="off", method="lexical", rerank=True, source="Example", limit=5
    )
    with search_snapshot(followup, settings, as_of=NOW, frozen_intent=intent) as (result, _):
        assert result.temporal.applied
        assert result.results[0].article_id == ids[29]
        assert result.temporal.intent.as_of == NOW


def test_frozen_research_intent_rejects_scope_drift(dated_articles):
    from backend.app.services.time_intent import resolve_time_intent

    settings, _ = dated_articles
    original = SearchRequest(q="NVIDIA today", method="lexical")
    intent = resolve_time_intent(original, NOW)
    with pytest.raises(ValueError, match="scope"):
        with search_snapshot(original, settings, as_of=NOW, frozen_intent=intent):
            pytest.fail("Mismatched dates must fail before opening retrieval dependencies")


@pytest.mark.parametrize("method", ["lexical", "hybrid"])
def test_historical_window_is_applied_before_retrieval(dated_articles, monkeypatch, method):
    settings, ids = dated_articles
    # For hybrid, inject an empty semantic branch while retaining real SQL lexical retrieval.
    if method == "hybrid":
        from unittest.mock import MagicMock

        from backend.app.schemas.search import SearchResponse
        from backend.app.services.semantic_search import SemanticSearchService

        monkeypatch.setattr(
            "backend.app.search_runtime.provider_for", lambda _: MagicMock(dimensions=2)
        )
        monkeypatch.setattr(
            "backend.app.search_runtime.open_store", lambda *args, **kwargs: MagicMock(dimensions=2)
        )
        monkeypatch.setattr(
            SemanticSearchService,
            "search",
            lambda self, request: SearchResponse(query=request.q, count=0, results=[]),
        )
    request = SearchRequest(
        q="NVIDIA shipment in January", method=method, rerank=True, limit=5, source="Example"
    )
    response = search(request, settings)
    assert response.temporal.intent.date_from.isoformat() == "2026-01-01"
    assert response.temporal.intent.retrieval_query == "NVIDIA shipment"
    assert not response.temporal.applied and response.query == request.q
    assert response.count == 5 and all(hit.published_at.month == 1 for hit in response.results)
    assert ids[29] not in [item.article_id for item in response.reranking.scores]
    assert all(hit.relevance_score == 6.0 for hit in response.results)


def test_api_recency_modes_and_explicit_dates(dated_articles, monkeypatch):
    settings, _ = dated_articles
    monkeypatch.setattr("backend.app.search_runtime.Settings", lambda: settings)
    with TestClient(create_app()) as client:
        result = client.get(
            "/search",
            params={
                "q": "NVIDIA shipment",
                "method": "lexical",
                "recency": "latest",
                "date_from": "2026-01-01",
                "date_to": "2026-01-31",
            },
        )
        assert result.status_code == 200
        assert result.json()["temporal"]["intent"]["kind"] == "explicit"
        assert not result.json()["temporal"]["applied"]
        assert client.get("/search", params={"q": "news", "recency": "invalid"}).status_code == 422
