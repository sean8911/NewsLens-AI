"""Real RSS/SQL/Qdrant/API wiring with deterministic neural-provider substitutes."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.catalog_runtime import CatalogRuntime, read_system_status
from backend.app.chat_runtime import run_chat
from backend.app.database.session import create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.services.article_embeddings import ArticleEmbeddingService
from backend.app.services.event_clustering import EventClusteringService
from backend.app.services.events import EventReadService
from backend.app.services.news_updates import NewsUpdateService
from ingestion.collector import RSSCollector
from ingestion.feeds.config import FeedConfig
from nlp.embeddings.store import open_store

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class Embeddings:
    version = "pipeline-test-embeddings"
    dimensions = 2

    def embed(self, text):
        return [0.0, 1.0] if "earnings" in text.lower() else [1.0, 0.0]


class Reranker:
    version = "pipeline-test-reranker"

    def score(self, query, documents):
        return [1.0 if "GPU" in document else -10.0 for document in documents]


class Selector:
    version = "pipeline-test-selector"

    def __init__(self):
        self.calls = 0
        self.invalid = False

    def generate(self, messages):
        self.calls += 1
        context = json.loads(messages[-1]["content"])["news_context"]
        first_per_article = {}
        for item in context:
            first_per_article.setdefault(item["article_id"], item["id"])
        # Reverse order: citation numbers must follow selection, not retrieval IDs.
        selected = list(reversed(first_per_article.values()))
        return json.dumps({"evidence_ids": ["E999"] if self.invalid else selected})


@pytest.fixture
def pipeline(database, tmp_path, monkeypatch):
    sessions = create_session_factory(database[0])
    settings = database[2].model_copy(update={"qdrant_path": tmp_path / "vectors"})
    feeds = [
        FeedConfig(
            id=name,
            source=f"{name.title()} Test News",
            url=f"https://{name}.example/rss",
            category="Technology",
        )
        for name in ("alpha", "beta")
    ]
    payloads = {
        str(feed.url): (FIXTURES / f"pipeline_{feed.id}.xml").read_bytes() for feed in feeds
    }

    class Fetcher:
        def fetch(self, url):
            return payloads[url]

    provider = Embeddings()
    store = open_store(settings, provider)
    try:
        updates = NewsUpdateService(
            sessions,
            feeds,
            RSSCollector(Fetcher()).collect,
            sync_embeddings=ArticleEmbeddingService(sessions, provider, store).sync,
            sync_events=EventClusteringService(sessions, provider, store).sync,
        )
        first, second = updates.run(), updates.run()
    finally:
        store.client.close()
    selector = Selector()
    monkeypatch.setattr("backend.app.search_runtime.Settings", lambda: settings)
    monkeypatch.setattr("backend.app.search_runtime.provider_for", lambda _: provider)
    monkeypatch.setattr("backend.app.search_runtime.reranker_for", lambda _: Reranker())
    monkeypatch.setattr("backend.app.chat_runtime.language_model_for", lambda _: selector)
    reader = EventReadService(sessions)
    app = create_app(
        chat=lambda request: run_chat(request, settings),
        events=reader.list,
        event_detail=reader.get,
        catalog=CatalogRuntime(settings),
        system_status=lambda: read_system_status(settings),
    )
    with TestClient(app) as api:
        yield api, sessions, first, second, selector


def test_rss_to_search_events_and_citations(pipeline):
    api, sessions, first, second, selector = pipeline
    assert first.succeeded and second.succeeded
    assert first.report.status == second.report.status == "succeeded"
    assert first.report.articles_found == 5
    assert first.report.inserted == 3
    assert first.report.duplicates == 1
    assert first.report.rejected == 1
    assert second.report.inserted == 0
    assert second.report.duplicates == 4
    with sessions() as session:
        records = {record.id: record for record in session.scalars(select(ArticleRecord))}
    assert len(records) == 3
    alpha = next(record for record in records.values() if record.url.endswith("alpha.example/gpu"))
    assert alpha.title == "NVIDIA launches a new GPU"
    assert alpha.description == "NVIDIA launched a new GPU for research & computing."
    assert all(
        record.cleaning_version and record.processing_status == "clean"
        for record in records.values()
    )

    params = {"q": "NVIDIA GPU", "method": "hybrid", "rerank": True, "recency": "off"}
    response = api.get("/search", params=params)
    assert response.status_code == 200
    search = response.json()
    assert search["fusion"]["lexical_candidates"] == 2
    assert search["reranking"]["model_version"] == Reranker.version
    assert {hit["article_id"] for hit in search["results"][:2]} == {
        record.id for record in records.values() if "GPU" in record.title
    }
    post = api.post("/search", json=params)
    assert post.status_code == 200
    assert post.json()["results"] == search["results"]

    response = api.get("/events")
    assert response.status_code == 200
    assert response.json()["count"] == 1
    event = response.json()["results"][0]
    assert event["article_count"] == event["source_count"] == 2
    detail = api.get(f"/events/{event['id']}")
    assert detail.status_code == 200
    assert {article["article_id"] for article in detail.json()["articles"]} == {
        hit["article_id"] for hit in search["results"][:2]
    }

    response = api.post(
        "/chat",
        json={
            "question": "NVIDIA GPU",
            "recency": "off",
            "answer_language": "en",
        },
    )
    assert response.status_code == 200
    answer = response.json()
    assert answer["status"] == "answered" and selector.calls == 1
    assert [source["article_id"] for source in answer["sources"]] == [
        hit["article_id"] for hit in reversed(search["results"][:2])
    ]
    for number, source in enumerate(answer["sources"], 1):
        record = records[source["article_id"]]
        assert source["citation_id"] == number
        assert source["url"] == record.url and source["source"] == record.source
        assert source["title"] == record.title
        assert datetime.fromisoformat(source["retrieved_at"]) == record.retrieved_at
        assert datetime.fromisoformat(source["published_at"]) == record.published_at
        assert source["excerpts"] == [record.title]
        assert f"“{record.title}” [{number}]" in answer["answer"]

    status = api.get("/system/status")
    assert status.status_code == 200
    assert status.json()["article_count"] == 3
    assert status.json()["last_successful_update_status"] == "succeeded"


def test_pipeline_empty_scope_skips_generation_and_unknown_evidence_is_rejected(pipeline):
    api, _, _, _, selector = pipeline
    response = api.post("/chat", json={"question": "NVIDIA GPU", "source": "Absent"})
    assert response.status_code == 200
    assert response.json()["reason"] == "no_evidence"
    assert response.json()["sources"] == [] and selector.calls == 0
    selector.invalid = True
    response = api.post("/chat", json={"question": "NVIDIA GPU", "recency": "off"})
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_information"
    assert response.json()["reason"] == "invalid_model_output"
    assert response.json()["sources"] == [] and selector.calls == 1
