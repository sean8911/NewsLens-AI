import pytest
from alembic import command
from sqlalchemy import select, text

from backend.app.database.session import create_session_factory
from backend.app.models.article_topic import ArticleTopicRecord
from backend.app.services.article_cleaning import ArticleCleaningService
from backend.app.services.article_storage import ArticleStorageService
from backend.app.services.article_topics import ArticleTopicService
from ingestion.cleaners.processor import ArticleProcessor
from ingestion.parsers.rss import normalize_entry
from nlp.classification.classifier import HYPOTHESES, TopicClassifier

pytestmark = pytest.mark.integration
TEXT = (
    "Researchers announced a new computer chip for artificial intelligence applications "
    "in technology companies."
)


class Backend:
    version = "fixture/v1"
    available = True
    fail = False

    def score(self, text, hypotheses):
        if self.fail:
            raise RuntimeError("Simulated model failure")
        scores = dict.fromkeys(HYPOTHESES, 0.1)
        scores.update({"Technology": 0.91, "Artificial Intelligence": 0.88})
        if "hospital" in text:
            scores.update({"Technology": 0.1, "Artificial Intelligence": 0.1, "Health": 0.95})
        return scores


def setup(database, feed, retrieved_at):
    sessions = create_session_factory(database[0])
    backend = Backend()
    classifier = TopicClassifier(backend)
    storage = ArticleStorageService(sessions, classifier)
    article = normalize_entry(
        {"title": "Research announcement", "summary": TEXT, "link": "https://example.com/story"},
        feed,
        retrieved_at,
    )
    return sessions, backend, classifier, storage, article


def test_topic_rows_scores_and_rss_category_are_separate(database, feed, retrieved_at):
    sessions, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    record = storage.list_recent()[0]
    assert record.category == feed.category
    assert [label.topic for label in record.article_topics] == [
        "Technology",
        "Artificial Intelligence",
    ]
    assert record.article_topics[0].confidence == 0.91
    assert record.topic_scores["Finance"] == 0.1
    audit = ArticleCleaningService(sessions).audit(article_id=record.id)[0]
    assert audit["clean"]["article_topics"][0]["topic"] == "Technology"


def test_duplicate_changes_use_retained_content_and_replace_labels(database, feed, retrieved_at):
    sessions, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    original = storage.list_recent()[0]
    enriched = ArticleProcessor().process(
        article,
        content=(
            "The hospital opened a new medical treatment center. "
            "Doctors provide healthcare and medicine to patients."
        ),
    )
    enriched.content_available = True
    storage.store_processed([enriched])
    stored = storage.get(original.id)
    assert [label.topic for label in stored.article_topics] == ["Health"]
    assert stored.topic_text_source == "title_content" and stored.description == TEXT
    storage.store([article])
    again = storage.get(original.id)
    assert again.updated_at == stored.updated_at
    with sessions() as session:
        assert len(list(session.scalars(select(ArticleTopicRecord)))) == 1


def test_failed_refresh_retains_previous_labels_with_audit(database, feed, retrieved_at):
    sessions, backend, classifier, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    backend.fail = True
    batch = ArticleTopicService(sessions, classifier).backfill_batch(refresh=True)
    assert batch.errors == 1
    after = storage.get(before.id)
    assert after.article_topics == before.article_topics
    assert after.updated_at == before.updated_at
    assert (
        "topic_failure_previous_labels_retained"
        in ArticleCleaningService(sessions).audit()[0]["issues"]
    )


def test_backfill_idempotence_and_force_refresh(database, feed, retrieved_at):
    sessions, _, classifier, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    with sessions.begin() as session:
        session.execute(text("UPDATE articles SET topic_status='not_attempted', topic_model=NULL"))
    service = ArticleTopicService(sessions, classifier)
    assert service.backfill_batch(1).classified == 1
    assert service.backfill_batch(1).processed == 0
    assert service.backfill_batch(1, refresh=True).classified == 1


def test_labels_and_metadata_roll_back_on_audit_failure(database, feed, retrieved_at, monkeypatch):
    from backend.app.repositories.processing import ProcessingRepository

    sessions, backend, classifier, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    classifier.version = "new-model-version"
    monkeypatch.setattr(
        ProcessingRepository, "add", lambda *_: (_ for _ in ()).throw(RuntimeError())
    )
    with pytest.raises(RuntimeError):
        ArticleTopicService(sessions, classifier).backfill_batch(refresh=True)
    after = storage.get(before.id)
    assert after.topic_model == before.topic_model and after.article_topics == before.article_topics


def test_migration_roundtrip_keeps_articles_and_language(database, feed, retrieved_at):
    _, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    before = storage.list_recent()[0]
    engine, config, _ = database
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.check(config)
        command.downgrade(config, "0005")
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 1
        command.upgrade(config, "head")
    after = storage.get(before.id)
    assert after.language == before.language and after.description == before.description
    assert after.article_topics == [] and after.topic_status == "not_attempted"


def test_topic_rows_cascade_and_confidence_constraint(database, feed, retrieved_at):
    from sqlalchemy.exc import DBAPIError

    sessions, _, _, storage, article = setup(database, feed, retrieved_at)
    storage.store([article])
    article_id = storage.list_recent()[0].id
    with pytest.raises(DBAPIError) as failure, sessions.begin() as session:
        session.add(
            ArticleTopicRecord(article_id=article_id, topic="Invalid confidence", confidence=2)
        )
        session.flush()
    assert failure.value.orig.args[0]["C"] == "23514"  # PostgreSQL check violation.
    with sessions.begin() as session:
        session.execute(text("DELETE FROM articles WHERE id=:id"), {"id": article_id})
    with sessions() as session:
        assert list(session.scalars(select(ArticleTopicRecord))) == []
