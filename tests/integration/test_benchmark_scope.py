from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import QdrantClient

from backend.app.database.session import create_session_factory
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchRequest
from backend.app.services.article_embeddings import ArticleEmbeddingService
from backend.app.services.lexical_search import LexicalSearchService
from backend.app.services.reranking import rerank_results
from backend.app.services.semantic_search import SemanticSearchService
from nlp.embeddings.store import QdrantEmbeddingStore

pytestmark = pytest.mark.integration


class Provider:
    version = "benchmark-test-provider"
    dimensions = 2

    def embed(self, text):
        return [1.0, 0.0]


class Reranker:
    version = "benchmark-test-reranker"

    def score(self, query, documents):
        return [float(i) for i in range(len(documents))]


def test_snapshot_runner_with_real_sql_and_qdrant(database, monkeypatch):
    from backend.app.evaluation import benchmark_runtime as runtime
    from backend.app.evaluation.news_dataset import load_dataset

    @contextmanager
    def isolated(settings):
        yield database[0]  # The fixture owns this unique migrated schema and its cleanup.

    monkeypatch.setattr(runtime, "evaluation_database", isolated)
    monkeypatch.setattr(runtime, "provider_for", lambda _: Provider())
    monkeypatch.setattr(runtime, "reranker_for", lambda _: Reranker())
    result = runtime.run_benchmark(
        load_dataset(), database[2], case_ids=["fact-xbox-limit", "absent-reuters-maths"],
    )
    assert result["status"] == "complete" and result["scope"] == "subset"
    assert result["provenance"]["indexing"]["indexed"] == 247
    for method, run in result["cases"][1]["runs"].items():
        assert run["metrics"] is None and run["article_ids"] == []
        assert result["aggregates"][method]["answerable_count"] == 1
    assert result["cases"][0]["runs"]["hybrid_reranked"]["reranking"] is not None


def test_future_same_day_articles_are_excluded_before_both_rankings(database):
    sessions = create_session_factory(database[0])
    now = datetime(2026, 9, 10, 4, tzinfo=UTC)
    with sessions.begin() as session:
        records = []
        for index, timestamp in enumerate([now + timedelta(seconds=1), now, None]):
            record = ArticleRecord(
                title="Orion launch report",
                description="Orion launch report details.",
                source="Example",
                category="Technology",
                language="en",
                url=f"https://example.com/benchmark/{index}",
                published_at=timestamp,
                retrieved_at=now,
                cleaning_version="1",
                processing_status="clean",
            )
            session.add(record)
            session.flush()
            records.append(record.id)
    client = QdrantClient(":memory:")
    try:
        store = QdrantEmbeddingStore(client, "benchmark_scope", 2)
        assert ArticleEmbeddingService(sessions, Provider(), store).sync()["indexed"] == 3
        request = SearchRequest(q="Orion", limit=1)
        lexical = LexicalSearchService(sessions, as_of=now).search(request)
        semantic = SemanticSearchService(sessions, Provider(), store, as_of=now).search(request)
        assert lexical.results[0].article_id != records[0]
        assert semantic.results[0].article_id != records[0]
        reranked = rerank_results(semantic, request, sessions, Reranker(), 10, as_of=now)
        assert reranked.results[0].article_id in records[1:]
        future = SemanticSearchService(sessions, Provider(), store).search(request)
        # A known future candidate may not be hydrated by an as-of-scoped reranker.
        if future.results[0].article_id == records[0]:
            with pytest.raises(RuntimeError, match="snapshot"):
                rerank_results(future, request, sessions, Reranker(), 10, as_of=now)
    finally:
        client.close()
