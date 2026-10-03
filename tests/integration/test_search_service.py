from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import select

from backend.app.database.session import create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.models.article_embedding import ArticleEmbeddingRecord
from backend.app.schemas.search import SearchRequest
from backend.app.services.article_embeddings import ArticleEmbeddingService
from backend.app.services.semantic_search import SemanticSearchService
from nlp.embeddings.store import QdrantEmbeddingStore
from nlp.summarization.summarizer import ArticleSummarizer

pytestmark = pytest.mark.integration


class Provider:
    dimensions = 2
    version = "search-test-v1"

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        if "Championship" in text:
            return [0.0, 1.0]
        if "Offices" in text:
            return [0.8, 0.2]
        return [1.0, 0.0]


@pytest.fixture
def search_setup(database, tmp_path):
    sessions = create_session_factory(database[0])
    with sessions.begin() as session:
        records = [
            ArticleRecord(
                title=title,
                description=description,
                source=source,
                category=category,
                language=language,
                published_at=published,
                retrieved_at=datetime.now(UTC),
                url=f"https://example.com/{i}",
                cleaning_version="1",
                processing_status="clean",
            )
            for i, (title, description, source, category, language, published) in enumerate(
                [
                    (
                        "Robots take routine factory tasks",
                        "Machines reshape the workforce.",
                        "Factory News",
                        "Technology",
                        "en",
                        datetime(2026, 9, 9, 23, 59, 59, 999999, UTC),
                    ),
                    (
                        "Championship final",
                        "A football club wins a trophy.",
                        "Sports News",
                        "Sports",
                        "ms",
                        datetime(2026, 9, 10, tzinfo=UTC),
                    ),
                    (
                        "Offices automate repetitive tasks",
                        "Businesses invest in automation.",
                        "Factory News",
                        "Business",
                        None,
                        None,
                    ),
                ]
            )
        ]
        session.add_all(records)
    # API tests use a worker thread. Access is serialized by create_app's lock.
    client = QdrantClient(path=str(tmp_path / "qdrant"), force_disable_check_same_thread=True)
    provider = Provider()
    store = QdrantEmbeddingStore(client, "search_test", 2)
    index = ArticleEmbeddingService(sessions, provider, store)
    assert index.sync()["indexed"] == 3
    service = SemanticSearchService(sessions, provider, store)
    try:
        yield sessions, provider, store, service, records, index
    finally:
        client.close()


def test_ranked_results_use_same_provider_and_postgres_fields(search_setup):
    sessions, provider, store, search, records, _ = search_setup
    with sessions() as session:
        before = [
            (r.input_hash, r.indexed_at) for r in session.scalars(select(ArticleEmbeddingRecord))
        ]
    result = search.search(SearchRequest(q="How is AI changing employment?", limit=2))
    assert [r.article_id for r in result.results] == [records[0].id, records[2].id]
    assert result.results[0].relevance_score > result.results[1].relevance_score
    assert result.results[0].title == records[0].title
    assert result.results[0].summary is None
    assert result.collection == store.collection and result.model_version == provider.version
    assert provider.calls[-1] == "How is AI changing employment?"
    with sessions() as session:
        assert before == [
            (r.input_hash, r.indexed_at) for r in session.scalars(select(ArticleEmbeddingRecord))
        ]


@pytest.mark.parametrize(
    "filters,expected",
    [
        ({"source": "Sports News"}, [1]),
        ({"category": "Business"}, [2]),
        ({"language": "en"}, [0]),
        ({"language": "unknown"}, [2]),
        ({"date_from": "2026-09-09", "date_to": "2026-09-09"}, [0]),
        ({"date_from": "2026-09-10"}, [1]),
        ({"date_to": "2026-09-09"}, [0]),
        (
            {
                "source": "Factory News",
                "language": "en",
                "category": "Technology",
                "date_from": "2026-09-09",
                "date_to": "2026-09-09",
            },
            [0],
        ),
        ({"source": "Missing"}, []),
        ({"source": "' OR 1=1 --"}, []),
    ],
)
def test_filters_apply_before_ranking(search_setup, filters, expected):
    _, _, _, search, records, _ = search_setup
    result = search.search(SearchRequest(q="employment", limit=1, **filters))
    assert [hit.article_id for hit in result.results] == [records[i].id for i in expected]


def test_current_sql_metadata_controls_membership_when_payload_is_stale(search_setup):
    sessions, _, store, search, records, _ = search_setup
    with sessions.begin() as session:
        record = session.get(ArticleRecord, records[0].id)
        record.source = "Renamed Publisher"
        record.category = "Science"
    assert store.get(records[0].id).payload["source"] == "Factory News"
    result = search.search(SearchRequest(q="work", source="Renamed Publisher", category="Science"))
    assert result.count == 1 and result.results[0].source == "Renamed Publisher"


@pytest.mark.parametrize(
    "change", ["deleted", "rejected", "content", "error", "model", "point", "payload"]
)
def test_stale_missing_and_ineligible_results_are_excluded(search_setup, change):
    sessions, _, store, search, records, _ = search_setup
    article_id = records[0].id
    with sessions.begin() as session:
        record = session.get(ArticleRecord, article_id)
        state = session.get(ArticleEmbeddingRecord, (article_id, store.collection))
        if change == "deleted":
            session.delete(record)
        elif change == "rejected":
            record.processing_status = "rejected"
        elif change == "content":
            record.content = "Changed retained article text."
        elif change == "error":
            state.status = "error"
        elif change == "model":
            state.model_version = "wrong-model"
    if change == "point":
        store.delete(article_id)
    if change == "payload":
        point = store.get(article_id)
        store.upsert(article_id, point.vector, {**point.payload, "input_hash": "stale"})
    result = search.search(SearchRequest(q="employment", limit=1))
    assert result.results[0].article_id == records[2].id


def test_api_returns_serialized_real_database_search(search_setup):
    _, _, _, search, records, _ = search_setup
    with TestClient(create_app(search.search)) as client:
        result = client.get("/search", params={"q": "employment", "limit": 1}).json()
    hit = result["results"][0]
    assert hit["article_id"] == records[0].id
    assert hit["published_at"].startswith("2026-09-09")
    assert set(hit) == {
        "article_id",
        "title",
        "source",
        "published_at",
        "relevance_score",
        "summary",
        "url",
        "language",
        "category",
    }


def test_search_fetches_next_page_after_stale_high_ranked_points(search_setup):
    sessions, _, store, search, records, index = search_setup
    with sessions.begin() as session:
        for i in range(55):
            session.add(
                ArticleRecord(
                    title=f"Robots {i}",
                    source="Extra",
                    category="Technology",
                    url=f"https://example.com/extra/{i}",
                    retrieved_at=datetime.now(UTC),
                    cleaning_version="1",
                    processing_status="clean",
                )
            )
    index.sync()
    for article_id in store.ids():
        if article_id != records[2].id:
            point = store.get(article_id)
            store.upsert(article_id, point.vector, {**point.payload, "model_version": "stale"})
    result = search.search(SearchRequest(q="employment", limit=1))
    assert result.count == 1 and result.results[0].article_id == records[2].id


def test_summary_returned_only_when_it_matches_current_source(search_setup):
    sessions, _, _, search, records, index = search_setup
    with sessions.begin() as session:
        record = session.get(ArticleRecord, records[0].id)
        record.content = (
            "Companies are retraining their factory workers. "
            "Managers expect automation to change several routine tasks next year. "
            "The training programme will continue for six months before an independent review. "
            "Researchers will publish the final results after interviewing all participants."
        )
        summary = ArticleSummarizer().summarize(
            record.title, record.description, record.content, record.language
        )
        assert summary.summary_status == "summarized"
        for name, value in summary.values().items():
            setattr(record, name, value)
    index.sync()
    assert search.search(SearchRequest(q="work", limit=1)).results[0].summary == summary.summary
    with sessions.begin() as session:
        session.get(ArticleRecord, records[0].id).content = "New source without that summary."
    index.sync()
    assert search.search(SearchRequest(q="work", limit=1)).results[0].summary is None
