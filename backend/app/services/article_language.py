from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.article import ArticleRecord
from backend.app.repositories.processing import ProcessingRepository
from backend.app.schemas.article import Article
from ingestion.cleaners.processor import ProcessingResult, snapshot
from nlp.language.detector import ArticleLanguageDetector, LanguageResult


def refresh_language(
    session: Session,
    record: ArticleRecord,
    detector: ArticleLanguageDetector,
    *,
    force: bool = False,
    audit_status: str = "language_updated",
) -> LanguageResult | None:
    """Caller holds the row lock. Classify the text actually retained in the database."""
    if not force and record.language is not None and record.language_detector is None:
        return None  # Preserve pre-existing labels with unknown provenance.
    _, _, digest = detector.prepare(record.title, record.description, record.content)
    if not force and (
        record.language_input_hash == digest
        and record.language_detector == detector.version
        and record.language_status != "error"
    ):
        return None
    raw = snapshot(
        {
            "format": "language_backfill",
            "record": {
                column.name: getattr(record, column.name) for column in record.__table__.columns
            },
        }
    )
    result = detector.detect(record.title, record.description, record.content)
    article = Article.model_validate({key: getattr(record, key) for key in Article.model_fields})
    audit = ProcessingResult(
        raw,
        article,
        content=record.content,
        content_available=record.content_available,
        content_extraction_method=record.content_extraction_method,
        content_extraction_status=record.content_extraction_status,
        language_result=result,
    )
    if result.language_status != "error" or record.language is None:
        for key, value in result.values().items():
            setattr(record, key, value)
        record.updated_at = func.now()
    else:
        audit.issues.append("language_error_previous_label_retained")
    ProcessingRepository(session).add(audit, audit_status, record.id)
    return result


@dataclass(frozen=True)
class LanguageBatch:
    processed: int = 0
    detected: int = 0
    unknown: int = 0
    errors: int = 0
    last_id: int = 0


class ArticleLanguageService:
    def __init__(
        self, sessions: sessionmaker[Session], detector: ArticleLanguageDetector | None = None
    ):
        self.sessions = sessions
        self.detector = detector or ArticleLanguageDetector()

    def backfill_batch(
        self, limit: int = 100, *, after_id: int = 0, refresh: bool = False
    ) -> LanguageBatch:
        if not 1 <= limit <= 1000 or after_id < 0:
            raise ValueError("Use batch size 1..1000 and nonnegative cursor")
        detected = unknown = errors = 0
        with self.sessions.begin() as session:
            query = select(ArticleRecord).where(
                ArticleRecord.id > after_id,
                ArticleRecord.cleaning_version.is_not(None),
                ArticleRecord.processing_status != "rejected",
            )
            if not refresh:
                query = query.where(
                    ArticleRecord.language_detector.is_(None), ArticleRecord.language.is_(None)
                )
            records = list(
                session.scalars(query.order_by(ArticleRecord.id).limit(limit).with_for_update())
            )
            for record in records:
                result = refresh_language(
                    session,
                    record,
                    self.detector,
                    force=refresh,
                    audit_status="lang_backfilled",
                )
                if result is not None:
                    detected += result.language is not None
                    unknown += result.language is None
                    errors += result.language_status == "error"
            last_id = records[-1].id if records else after_id
        return LanguageBatch(len(records), detected, unknown, errors, last_id)
