from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from backend.app.database.session import create_session_factory
from backend.app.evaluation.cli import evaluate
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import FusionConfig, SearchRequest
from backend.app.search_runtime import run_search
from backend.app.services.article_embeddings import ArticleEmbeddingService
from backend.app.services.lexical_search import LexicalSearchService
from backend.app.services.semantic_search import SemanticSearchService
from nlp.embeddings.store import open_store

pytestmark = pytest.mark.integration


class Provider:
    version = "hybrid-test-model"
    dimensions = 2

    def embed(self, text):
        return [1.0, 0.0]


@pytest.fixture
def hybrid_setup(database, tmp_path, monkeypatch):
    settings = database[2].model_copy(update={"qdrant_path": tmp_path / "hybrid-qdrant"})
    sessions = create_session_factory(database[0])
    with sessions.begin() as session:
        record = ArticleRecord(
            title="Orion test report",
            source="Example",
            category="Technology",
            language="en",
            retrieved_at=datetime.now(UTC),
            published_at=datetime(2026, 9, 1, tzinfo=UTC),
            url="https://example.com/orion",
            cleaning_version="1",
            processing_status="clean",
        )
        session.add(record)
    store = open_store(settings, Provider())
    try:
        assert ArticleEmbeddingService(sessions, Provider(), store).sync()["indexed"] == 1
    finally:
        store.client.close()
    monkeypatch.setattr("backend.app.search_runtime.Settings", lambda: settings)
    monkeypatch.setattr("backend.app.search_runtime.provider_for", lambda _: Provider())
    return sessions, record.id


def test_real_api_fuses_current_results_and_preserves_filters(hybrid_setup):
    with TestClient(create_app()) as client:
        response = client.get(
            "/search",
            params={
                "q": "Orion",
                "method": "hybrid",
                "source": "Example",
                "category": "Technology",
                "language": "en",
                "date_from": "2026-09-01",
                "date_to": "2026-09-01",
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["method"] == "hybrid" and result["count"] == 1
        assert result["fusion"]["explanations"][0]["semantic"]["rank"] == 1
        assert result["fusion"]["explanations"][0]["lexical"]["rank"] == 1
        assert (
            client.get(
                "/search", params={"q": "Orion", "method": "hybrid", "source": "Other"}
            ).json()["count"]
            == 0
        )


def test_runtime_uses_one_repeatable_sql_snapshot(hybrid_setup, monkeypatch):
    sessions, article_id = hybrid_setup
    original = SemanticSearchService.search

    def update_between_branches(self, request):
        result = original(self, request)
        with sessions.begin() as writer:
            writer.get(ArticleRecord, article_id).title = "Changed title"
        return result

    monkeypatch.setattr(SemanticSearchService, "search", update_between_branches)
    result = run_search(SearchRequest(q="Orion", method="hybrid"))
    assert result.results[0].title == "Orion test report"
    assert result.fusion.explanations[0].lexical is not None
    with sessions() as session:
        assert session.get(ArticleRecord, article_id).title == "Changed title"


def test_hybrid_preserves_lexical_only_unembedded_articles(hybrid_setup):
    sessions, _ = hybrid_setup
    with sessions.begin() as session:
        record = session.scalar(select(ArticleRecord))
        record.content = "Orion body changed after embedding."
    result = run_search(SearchRequest(q="Orion", method="hybrid"))
    assert result.count == 1
    assert result.fusion.semantic_candidates == 0 and result.fusion.lexical_candidates == 1


def test_second_branch_failure_propagates(hybrid_setup, monkeypatch):
    def fail(*args):
        raise RuntimeError("outage")

    monkeypatch.setattr(LexicalSearchService, "search", fail)
    with pytest.raises(RuntimeError):
        run_search(SearchRequest(q="Orion", method="hybrid"))


def test_lexical_evaluation_is_isolated_and_needs_no_model(database, monkeypatch):
    settings = database[2].model_copy(update={"test_database_url": database[2].database_url})

    def forbidden(*args):
        pytest.fail("Lexical-only evaluation should not load models")

    monkeypatch.setattr("backend.app.evaluation.cli.provider_for", forbidden)
    dataset = {
        "name": "Test",
        "note": "Synthetic",
        "documents": [{"id": 1, "title": "Orion"}],
        "queries": [{"id": "one", "q": "Orion", "relevance": {"1": 3}}],
    }
    with database[0].connect() as connection:
        before = set(
            connection.scalars(
                text("SELECT nspname FROM pg_namespace WHERE nspname LIKE 'newslens_eval_%'")
            )
        )

    report = evaluate(dataset, settings, {"hybrid": FusionConfig()}, ["lexical"], 3)
    assert report["mean_metrics"]["lexical"]["ndcg"] == 1
    assert report["model_version"] is None
    with database[0].connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 0
        assert (
            set(
                connection.scalars(
                    text("SELECT nspname FROM pg_namespace WHERE nspname LIKE 'newslens_eval_%'")
                )
            )
            == before
        )


@pytest.mark.parametrize("method", ["semantic", "lexical", "hybrid"])
def test_reranking_is_optional_and_preserves_sql_snapshot(hybrid_setup, monkeypatch, method):
    sessions, article_id = hybrid_setup
    captured = []

    class Reranker:
        version = "reranking-test"

        def score(self, query, documents):
            captured.extend(documents)
            return [4.5] * len(documents)

    def forbidden(*args):
        pytest.fail("Disabled reranking must not construct a model")

    monkeypatch.setattr("backend.app.search_runtime.reranker_for", forbidden)
    assert run_search(SearchRequest(q="Orion", method=method)).reranking is None
    monkeypatch.setattr("backend.app.search_runtime.reranker_for", lambda _: Reranker())
    service = LexicalSearchService if method == "lexical" else SemanticSearchService
    original = service.search

    def change_after_retrieval(self, request):
        result = original(self, request)
        with sessions.begin() as writer:
            writer.get(ArticleRecord, article_id).title = "Changed concurrently"
        return result

    monkeypatch.setattr(service, "search", change_after_retrieval)
    response = run_search(SearchRequest(q="Orion", method=method, rerank=True))
    assert captured == ["Orion test report"]
    assert response.results[0].relevance_score == 4.5
    assert response.reranking.scores[0].retrieval_rank == 1
    if method == "hybrid":
        assert response.fusion.explanations[0].article_id == article_id


@pytest.mark.parametrize("limit, expected_pool, expected_count", [(5, 30, 5), (50, 35, 35)])
def test_reranking_candidate_depth_and_filters(
    database, monkeypatch, limit, expected_pool, expected_count
):
    settings = database[2]
    sessions = create_session_factory(database[0])
    with sessions.begin() as session:
        for index in range(40):
            session.add(
                ArticleRecord(
                    title=f"Orion report {index}",
                    content=f"Unique body {index}",
                    source="Example" if index < 35 else "Excluded",
                    category="Technology",
                    language="en",
                    published_at=datetime(2026, 9, 1, tzinfo=UTC),
                    retrieved_at=datetime.now(UTC),
                    url=f"https://example.com/pool/{index}",
                    cleaning_version="1",
                    processing_status="clean",
                )
            )
    captured = []

    class Reranker:
        version = "pool-test"

        def score(self, query, documents):
            captured.extend(documents)
            return list(range(len(documents)))

    monkeypatch.setattr("backend.app.search_runtime.Settings", lambda: settings)
    monkeypatch.setattr("backend.app.search_runtime.reranker_for", lambda _: Reranker())

    def forbidden(*args):
        pytest.fail("Lexical reranking must not need embedding models or Qdrant")

    monkeypatch.setattr("backend.app.search_runtime.provider_for", forbidden)
    with TestClient(create_app()) as client:
        response = client.get(
            "/search",
            params={
                "q": "Orion",
                "method": "lexical",
                "rerank": "true",
                "limit": limit,
                "source": "Example",
                "language": "en",
                "category": "Technology",
                "date_from": "2026-09-01",
                "date_to": "2026-09-01",
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert len(captured) == result["reranking"]["candidate_count"] == expected_pool
    assert result["count"] == expected_count
    assert all(hit["source"] == "Example" for hit in result["results"])
    assert result["reranking"]["scores"][-1]["reranking_rank"] == 1
    assert result["results"][0]["article_id"] == result["reranking"]["scores"][-1]["article_id"]


def test_evaluation_compares_reranking_and_retains_baseline(database, monkeypatch):
    settings = database[2].model_copy(update={"test_database_url": database[2].database_url})

    class Reranker:
        version = "evaluation-test"

        def score(self, query, documents):
            return [10.0 if "Relevant" in document else -10.0 for document in documents]

    monkeypatch.setattr("backend.app.evaluation.cli.reranker_for", lambda _: Reranker())
    dataset = {
        "name": "Test",
        "note": "Synthetic",
        "documents": [
            {"id": 1, "title": "Orion Orion Orion"},
            {"id": 2, "title": "Relevant Orion article"},
        ],
        "queries": [{"id": "one", "q": "Orion", "relevance": {"2": 3}}],
    }
    report = evaluate(
        dataset, settings, {"hybrid": FusionConfig()}, ["lexical"], 1, compare_reranking=True
    )
    assert report["mean_metrics"]["lexical"]["ndcg"] == 0
    assert report["mean_metrics"]["lexical_reranked"]["ndcg"] == 1
    assert report["reranking_ndcg_deltas"]["lexical"] == 1
    runs = report["cases"][0]["runs"]
    assert runs["lexical"]["response"]["reranking"] is None
    assert runs["lexical_reranked"]["response"]["reranking"]["candidate_count"] == 2


@pytest.mark.parametrize("case", ["answer", "empty", "low_score"])
def test_chat_uses_retrieval_and_releases_handles_before_llm(hybrid_setup, monkeypatch, case):
    from backend.app import search_runtime
    from backend.app.chat_runtime import run_chat
    from backend.app.schemas.chat import ChatRequest

    settings = search_runtime.Settings()
    monkeypatch.setattr("backend.app.chat_runtime.Settings", lambda: settings)

    class Reranker:
        version = "chat-test-reranker"

        def score(self, query, documents):
            floor = settings.multilingual_chat_min_reranking_score
            return [floor - 1 if case == "low_score" else 3.0] * len(documents)

    monkeypatch.setattr("backend.app.search_runtime.reranker_for", lambda _: Reranker())
    calls = []

    class LLM:
        version = "chat-test-llm"

        def generate(self, messages):
            calls.append(messages)
            # Acquiring the local collection again proves retrieval released its handle.
            store = open_store(settings, Provider(), create=False)
            store.client.close()
            return '{"evidence_ids": ["E1"]}'

    def factory(settings):
        assert case == "answer", "Do not load the LLM without relevant evidence"
        return LLM()

    monkeypatch.setattr("backend.app.chat_runtime.language_model_for", factory)
    request = ChatRequest(
        question="What happened to Orion?",
        source="Excluded" if case == "empty" else "Example",
        language="en",
        category="Technology",
        date_from="2026-09-01",
        date_to="2026-09-01",
    )
    result = run_chat(request)
    if case == "answer":
        assert result.status == "answered" and result.sources[0].article_id == hybrid_setup[1]
        assert result.sources[0].excerpts == ["Orion test report"] and len(calls) == 1
    else:
        assert result.reason == "no_evidence" and result.sources == [] and calls == []


def test_chat_evidence_uses_retrieval_snapshot_during_concurrent_update(hybrid_setup, monkeypatch):
    from backend.app import search_runtime

    settings = search_runtime.Settings()
    monkeypatch.setattr("backend.app.chat_runtime.Settings", lambda: settings)
    sessions, article_id = hybrid_setup
    original = SemanticSearchService.search

    def retrieve_and_update(self, request):
        result = original(self, request)
        with sessions.begin() as writer:
            writer.get(ArticleRecord, article_id).title = "Concurrent replacement story"
        return result

    monkeypatch.setattr(SemanticSearchService, "search", retrieve_and_update)

    class Reranker:
        version = "test"

        def score(self, query, documents):
            return [3.0] * len(documents)

    class LLM:
        version = "test"

        def generate(self, messages):
            assert "Orion test report" in messages[1]["content"]
            assert "Concurrent replacement story" not in messages[1]["content"]
            return '{"evidence_ids": ["E1"]}'

    monkeypatch.setattr("backend.app.search_runtime.reranker_for", lambda _: Reranker())
    monkeypatch.setattr("backend.app.chat_runtime.language_model_for", lambda _: LLM())
    with TestClient(create_app()) as client:
        response = client.post("/chat", json={"question": "Orion news?"})
    assert response.status_code == 200
    assert response.json()["sources"][0]["title"] == "Orion test report"
