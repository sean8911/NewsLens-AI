from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.article import ArticleRecord
from backend.app.repositories.entities import EntityRepository
from backend.app.repositories.processing import ProcessingRepository
from backend.app.schemas.article import Article
from ingestion.cleaners.processor import ProcessingResult, snapshot
from nlp.entities.extractor import EntityExtractor, EntityResult


def refresh_entities(
    session: Session, record: ArticleRecord, extractor: EntityExtractor, *, force: bool = False
) -> EntityResult | None:
    """Caller holds the article row lock; use its retained content and language."""
    _, _, digest, _ = extractor.prepare(
        record.title, record.description, record.content, record.language
    )
    same = record.entity_input_hash == digest and record.entity_model == extractor.version
    if not force and same:
        if record.entity_status not in {"error", "model_unavailable"}:
            return None
        if record.entity_status == "model_unavailable" and not extractor.backend.available:
            return None
    raw = snapshot(
        {
            "format": "stored_article_entities",
            "record": {
                column.name: getattr(record, column.name) for column in record.__table__.columns
            },
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "canonical_name": e.canonical_name,
                    "entity_type": e.entity_type,
                }
                for e in record.entities
            ],
        }
    )
    result = extractor.extract(record.title, record.description, record.content, record.language)
    article = Article.model_validate({key: getattr(record, key) for key in Article.model_fields})
    audit = ProcessingResult(
        raw,
        article,
        content=record.content,
        content_available=record.content_available,
        content_extraction_method=record.content_extraction_method,
        content_extraction_status=record.content_extraction_status,
        entity_result=result,
    )
    if result.entity_status in {"error", "model_unavailable"} and record.entity_status in {
        "extracted",
        "no_entities",
    }:
        audit.issues.append("entity_failure_previous_result_retained")
    else:
        for name, value in result.metadata().items():
            setattr(record, name, value)
        EntityRepository(session).replace(record.id, result)
        session.expire(record, ["entities"])
        record.updated_at = func.now()
    ProcessingRepository(session).add(audit, "entities_updated", record.id)
    return result


@dataclass(frozen=True)
class EntityBatch:
    processed: int = 0
    extracted: int = 0
    no_entities: int = 0
    skipped: int = 0
    errors: int = 0
    last_id: int = 0


class ArticleEntityService:
    def __init__(self, sessions: sessionmaker[Session], extractor: EntityExtractor | None = None):
        self.sessions = sessions
        self.extractor = extractor or EntityExtractor()

    def backfill_batch(
        self, limit: int = 50, *, after_id: int = 0, refresh: bool = False
    ) -> EntityBatch:
        if not 1 <= limit <= 100 or after_id < 0:
            raise ValueError("Use entity batch size 1..100 and a nonnegative cursor")
        counts = dict(extracted=0, no_entities=0, skipped=0, errors=0)
        with self.sessions.begin() as session:
            query = select(ArticleRecord).where(
                ArticleRecord.id > after_id,
                ArticleRecord.cleaning_version.is_not(None),
                ArticleRecord.processing_status != "rejected",
            )
            if not refresh:
                query = query.where(
                    ArticleRecord.entity_status.in_(["not_attempted", "error", "model_unavailable"])
                )
            records = list(
                session.scalars(query.order_by(ArticleRecord.id).limit(limit).with_for_update())
            )
            for record in records:
                result = refresh_entities(session, record, self.extractor, force=True)
                status = result.entity_status
                key = (
                    status
                    if status in {"extracted", "no_entities"}
                    else ("errors" if status in {"error", "model_unavailable"} else "skipped")
                )
                counts[key] += 1
            last_id = records[-1].id if records else after_id
        return EntityBatch(processed=len(records), last_id=last_id, **counts)
