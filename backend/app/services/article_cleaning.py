import json
from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.repositories.articles import ArticleRepository
from backend.app.repositories.processing import ProcessingRepository
from backend.app.schemas.article import Article
from ingestion.cleaners.processor import (
    CLEANING_VERSION,
    ArticleProcessor,
    ProcessingResult,
    snapshot,
)


@dataclass(frozen=True)
class BackfillResult:
    processed: int = 0
    cleaned: int = 0
    rejected: int = 0
    url_collisions: int = 0


class ArticleCleaningService:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    def backfill_batch(self, limit: int = 100) -> BackfillResult:
        if not 1 <= limit <= 1000:
            raise ValueError("batch size must be 1..1000")
        cleaned = rejected = collisions = 0
        with self.sessions.begin() as session:
            repository = ArticleRepository(session)
            audit = ProcessingRepository(session)
            records = repository.pending_cleaning(limit)
            for record in records:
                original = {
                    column.name: getattr(record, column.name) for column in record.__table__.columns
                }
                raw_json = snapshot({"format": "stored_article", "record": original})
                try:
                    article = Article.model_validate(
                        {key: original[key] for key in Article.model_fields}
                    )
                    result = ArticleProcessor().process(
                        article, content=record.content, raw_json=raw_json
                    )
                except ValidationError:
                    result = ProcessingResult(raw_json, None, ["legacy_validation_failed"])
                if result.article is None:
                    rejected += 1
                    status = "rejected"
                    record.processing_status = "rejected"
                else:
                    # Typed timestamps must remain datetimes for database writes.
                    values = result.article.model_dump() | {"url": str(result.article.url)}
                    existing = repository.get_by_url(values["url"])
                    if existing is not None and existing.id != record.id:
                        values.pop("url")
                        collisions += 1
                        status = "url_collision"
                        result.issues.append("url_collision_original_retained")
                        record.processing_status = "review"
                    else:
                        status = "backfilled"
                        record.processing_status = "clean"
                    for key, value in values.items():
                        setattr(record, key, value)
                    record.content = result.content
                    cleaned += 1
                record.cleaning_version = CLEANING_VERSION
                record.cleaning_issues = result.issues
                audit.add(result, status, record.id)
                session.flush()
        return BackfillResult(len(records), cleaned, rejected, collisions)

    def audit(
        self, *, article_id: int | None = None, status: str | None = None, limit: int = 20
    ) -> list[dict]:
        with self.sessions() as session:
            return [
                {
                    "id": record.id,
                    "article_id": record.article_id,
                    "status": record.status,
                    "issues": record.issues,
                    "cleaning_version": record.cleaning_version,
                    "created_at": record.created_at.isoformat(),
                    "raw": json.loads(record.raw_json),
                    "clean": json.loads(record.clean_json) if record.clean_json else None,
                }
                for record in ProcessingRepository(session).list_recent(
                    article_id=article_id, status=status, limit=limit
                )
            ]
