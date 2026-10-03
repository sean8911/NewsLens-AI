from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.article import ArticleRecord
from backend.app.repositories.processing import ProcessingRepository
from backend.app.repositories.topics import TopicRepository
from backend.app.schemas.article import Article
from ingestion.cleaners.processor import ProcessingResult, snapshot
from nlp.classification.classifier import TopicClassifier, TopicResult


def refresh_topics(
    session: Session, record: ArticleRecord, classifier: TopicClassifier, *, force: bool = False
) -> TopicResult | None:
    """Caller holds the article row lock, including when content/language were updated."""
    _, _, digest = classifier.prepare(
        record.title, record.description, record.content, record.language
    )
    same = record.topic_input_hash == digest and record.topic_model == classifier.version
    if not force and same:
        if record.topic_status not in {"error", "model_unavailable"}:
            return None
        if record.topic_status == "model_unavailable" and not classifier.backend.available:
            return None
    raw = snapshot(
        {
            "format": "stored_article_topics",
            "record": {
                column.name: getattr(record, column.name) for column in record.__table__.columns
            },
            "article_topics": [
                {"topic": label.topic, "confidence": label.confidence}
                for label in record.article_topics
            ],
        }
    )
    result = classifier.classify(record.title, record.description, record.content, record.language)
    article = Article.model_validate({key: getattr(record, key) for key in Article.model_fields})
    audit = ProcessingResult(
        raw,
        article,
        content=record.content,
        content_available=record.content_available,
        content_extraction_method=record.content_extraction_method,
        content_extraction_status=record.content_extraction_status,
        topic_result=result,
    )
    if result.topic_status in {"error", "model_unavailable"} and record.article_topics:
        audit.issues.append("topic_failure_previous_labels_retained")
    else:
        for name, value in result.metadata().items():
            setattr(record, name, value)
        TopicRepository(session).replace(record.id, result)
        session.expire(record, ["article_topics"])
        record.updated_at = func.now()
    ProcessingRepository(session).add(audit, "topics_updated", record.id)
    return result


@dataclass(frozen=True)
class TopicBatch:
    processed: int = 0
    classified: int = 0
    other: int = 0
    skipped: int = 0
    errors: int = 0
    last_id: int = 0


class ArticleTopicService:
    def __init__(self, sessions: sessionmaker[Session], classifier: TopicClassifier | None = None):
        self.sessions = sessions
        self.classifier = classifier or TopicClassifier()

    def backfill_batch(
        self, limit: int = 10, *, after_id: int = 0, refresh: bool = False
    ) -> TopicBatch:
        if not 1 <= limit <= 100 or after_id < 0:
            raise ValueError("Use topic batch size 1..100 and a nonnegative cursor")
        counts = dict(classified=0, other=0, skipped=0, errors=0)
        with self.sessions.begin() as session:
            query = select(ArticleRecord).where(
                ArticleRecord.id > after_id,
                ArticleRecord.cleaning_version.is_not(None),
                ArticleRecord.processing_status != "rejected",
            )
            if not refresh:
                query = query.where(
                    ArticleRecord.topic_status.in_(["not_attempted", "error", "model_unavailable"])
                )
            records = list(
                session.scalars(query.order_by(ArticleRecord.id).limit(limit).with_for_update())
            )
            for record in records:
                result = refresh_topics(session, record, self.classifier, force=True)
                status = result.topic_status
                key = (
                    status
                    if status in {"classified", "other"}
                    else ("errors" if status in {"error", "model_unavailable"} else "skipped")
                )
                counts[key] += 1
            last_id = records[-1].id if records else after_id
        return TopicBatch(processed=len(records), last_id=last_id, **counts)
