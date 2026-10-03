import pytest
from alembic import command
from qdrant_client import QdrantClient
from sqlalchemy import event, select, text
from sqlalchemy.orm import Session

from backend.app.database.session import create_session_factory
from backend.app.models.article import ArticleRecord
from backend.app.models.article_embedding import ArticleEmbeddingRecord
from backend.app.services.article_embeddings import ArticleEmbeddingService, article_input
from backend.app.services.article_storage import ArticleStorageService
from backend.app.services.news_updates import NewsUpdateService, read_update_status
from ingestion.collector import RSSCollector
from ingestion.parsers.rss import normalize_entry
from nlp.embeddings.store import QdrantEmbeddingStore

pytestmark = pytest.mark.integration
BODY = (
    "Researchers reported a new technology study. The team tested a faster computer chip. "
    "Engineers will collect more measurements before publishing detailed results next year. "
    "The experiment will continue at the university laboratory for another six months."
)


class Provider:
    version = "test-model-v1"
    dimensions = 2

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [3.0, 4.0]


@pytest.fixture
def indexed_setup(database, feed, retrieved_at, tmp_path):
    sessions = create_session_factory(database[0])
    storage = ArticleStorageService(sessions)
    article = normalize_entry(
        {"title": "Technology research", "summary": BODY, "link": "https://example.com/story"},
        feed,
        retrieved_at,
    )
    storage.store([article])
    article_id = storage.list_recent()[0].id
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    provider = Provider()
    store = QdrantEmbeddingStore(client, "test_articles", provider.dimensions)
    service = ArticleEmbeddingService(sessions, provider, store)
    try:
        yield sessions, provider, store, service, article_id
    finally:
        client.close()


def test_backfill_idempotence_metadata_and_missing_point_repair(indexed_setup):
    sessions, provider, store, service, article_id = indexed_setup
    assert service.sync()["indexed"] == 1
    point = store.get(article_id)
    assert point.payload["article_id"] == article_id
    assert point.payload["source"] == "Example News"
    with sessions() as session:
        before = session.get(ArticleEmbeddingRecord, (article_id, store.collection)).indexed_at
        updated = session.get(ArticleRecord, article_id).updated_at
    assert service.sync()["unchanged"] == 1 and len(provider.calls) == 1
    with sessions.begin() as session:
        record = session.get(ArticleRecord, article_id)
        assert record.updated_at == updated
        assert (
            session.get(ArticleEmbeddingRecord, (article_id, store.collection)).indexed_at == before
        )
        record.category = "Science"
    assert service.sync()["indexed"] == 1 and len(provider.calls) == 1
    assert store.get(article_id).payload["category"] == "Science"
    store.delete(article_id)
    assert service.sync()["indexed"] == 1 and len(provider.calls) == 2
    assert service.sync(refresh=True)["indexed"] == 1 and len(provider.calls) == 3


def test_content_refresh_uses_retained_sql_data_and_excludes_stale_summary(indexed_setup):
    sessions, provider, store, service, article_id = indexed_setup
    service.sync()
    with sessions.begin() as session:
        record = session.get(ArticleRecord, article_id)
        assert record.summary_status == "summarized"
        assert "summary" in article_input(record).fields
        record.content = "A completely different report about international football."
        assert "summary" not in article_input(record).fields
    assert service.sync()["indexed"] == 1
    assert "international football" in provider.calls[-1]
    assert store.get(article_id).payload["input_fields"] == ["title", "description", "content"]


def test_outage_preserves_article_and_successful_vector_then_retries(indexed_setup, monkeypatch):
    sessions, _, store, service, article_id = indexed_setup
    service.sync()
    before = store.get(article_id)
    with monkeypatch.context() as patch:
        patch.setattr(store, "upsert", lambda *args: (_ for _ in ()).throw(RuntimeError("secret")))
        assert service.sync(refresh=True)["errors"] == 1
    assert store.get(article_id) == before
    with sessions() as session:
        state = session.get(ArticleEmbeddingRecord, (article_id, store.collection))
        assert state.status == "error" and state.indexed_at is not None
        assert state.error_type == "RuntimeError"
        assert session.get(ArticleRecord, article_id) is not None
    assert service.sync()["indexed"] == 1


def test_sql_commit_failure_after_qdrant_upsert_reconciles(indexed_setup):
    sessions, provider, store, service, article_id = indexed_setup

    def crash(session):
        raise RuntimeError("Simulated SQL commit failure")

    event.listen(Session, "before_commit", crash)
    try:
        with pytest.raises(RuntimeError, match="SQL commit"):
            service.sync()
    finally:
        event.remove(Session, "before_commit", crash)
    assert store.get(article_id) is not None
    with sessions() as session:
        assert session.get(ArticleEmbeddingRecord, (article_id, store.collection)) is None
    assert service.sync()["indexed"] == 1 and len(provider.calls) == 1


@pytest.mark.parametrize("deleted", [False, True])
def test_prune_rejected_or_deleted_articles(indexed_setup, deleted):
    sessions, _, store, service, article_id = indexed_setup
    service.sync()
    with sessions.begin() as session:
        record = session.get(ArticleRecord, article_id)
        if deleted:
            session.delete(record)
        else:
            record.processing_status = "rejected"
    assert service.sync()["removed"] == 1
    assert store.get(article_id) is None
    with sessions() as session:
        assert session.get(ArticleEmbeddingRecord, (article_id, store.collection)) is None


def test_model_regeneration_keeps_independent_sql_ledgers(indexed_setup):
    sessions, _, store, service, article_id = indexed_setup
    service.sync()
    new_provider = Provider()
    new_provider.version = "test-model-v2"
    new_store = QdrantEmbeddingStore(store.client, "new_model", 2)
    other = ArticleEmbeddingService(sessions, new_provider, new_store)
    assert other.sync()["indexed"] == 1
    assert store.get(article_id) is not None and new_store.get(article_id) is not None
    with sessions() as session:
        states = list(session.scalars(select(ArticleEmbeddingRecord)))
        assert len(states) == 2
        assert {s.model_version for s in states} == {"test-model-v1", "test-model-v2"}


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_embedding_hook_runs_after_commit_and_failure_keeps_ingestion_success(
    database, feed, rss_bytes, failure, caplog
):
    sessions = create_session_factory(database[0])

    class Fetcher:
        def fetch(self, url):
            return rss_bytes

    def sync():
        assert len(ArticleStorageService(sessions).list_recent()) == 3
        assert read_update_status(sessions).recent_runs[0].status == "succeeded"
        raise failure("sensitive endpoint")

    service = NewsUpdateService(
        sessions, [feed], RSSCollector(Fetcher()).collect, sync_embeddings=sync
    )
    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            service.run()
    else:
        assert service.run().succeeded
    assert read_update_status(sessions).recent_runs[0].status == "succeeded"
    assert "sensitive endpoint" not in caplog.text


def test_migration_roundtrip_preserves_articles(indexed_setup, database):
    _, _, _, service, _ = indexed_setup
    service.sync()
    engine, config, _ = database
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "0009")
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 1
        command.upgrade(config, "head")
        assert connection.scalar(text("SELECT count(*) FROM article_embeddings")) == 0
    assert service.sync()["indexed"] == 1


def test_initial_provider_failure_retries_and_unknown_language_is_eligible(
    indexed_setup, monkeypatch
):
    sessions, provider, store, service, article_id = indexed_setup
    with sessions.begin() as session:
        session.get(ArticleRecord, article_id).language = None
    with monkeypatch.context() as patch:
        patch.setattr(provider, "embed", lambda *_: (_ for _ in ()).throw(RuntimeError()))
        assert service.sync()["errors"] == 1
    assert store.get(article_id) is None
    assert service.sync()["indexed"] == 1
    assert store.get(article_id).payload["language"] is None


def test_batch_cursor_visits_each_article_after_one_failure(indexed_setup, monkeypatch):
    sessions, provider, _, service, article_id = indexed_setup
    with sessions.begin() as session:
        original = session.get(ArticleRecord, article_id)
        session.add(
            ArticleRecord(
                title="Second headline",
                description="Another news story.",
                source=original.source,
                url="https://example.com/second",
                category=original.category,
                retrieved_at=original.retrieved_at,
                cleaning_version="1",
                processing_status="clean",
            )
        )
    original_embed = provider.embed

    def fail_first(text):
        if "Technology research" in text:
            raise RuntimeError()
        return original_embed(text)

    monkeypatch.setattr(provider, "embed", fail_first)
    result = service.sync(batch_size=1)
    assert (result["processed"], result["indexed"], result["errors"]) == (2, 1, 1)
