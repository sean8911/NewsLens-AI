import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.performance import monitored
from backend.app.models.article import ArticleRecord
from backend.app.models.article_embedding import ArticleEmbeddingRecord
from nlp.embeddings.service import EmbeddingService, digest, normalized, prepare
from nlp.embeddings.store import QdrantEmbeddingStore
from nlp.summarization.summarizer import ArticleSummarizer

logger = logging.getLogger(__name__)


def current_summary(record: ArticleRecord) -> str | None:
    # A retained summary after a failed refresh can describe older content. Exclude it.
    _, _, summary_hash, _ = ArticleSummarizer().prepare(
        record.title, record.description, record.content, record.language
    )
    return (
        record.summary
        if (record.summary_status == "summarized" and record.summary_input_hash == summary_hash)
        else None
    )


def article_input(record: ArticleRecord):
    return prepare(record.title, record.description, record.content, current_summary(record))


def eligible(record: ArticleRecord | None) -> bool:
    return bool(record and record.cleaning_version and record.processing_status != "rejected")


@dataclass
class EmbeddingBatch:
    processed: int = 0
    indexed: int = 0
    unchanged: int = 0
    errors: int = 0
    removed: int = 0
    last_id: int = 0


class ArticleEmbeddingService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        provider: EmbeddingService,
        store: QdrantEmbeddingStore,
    ):
        self.sessions, self.provider, self.store = sessions, provider, store

    def backfill_batch(
        self, limit: int = 50, *, after_id: int = 0, refresh: bool = False
    ) -> EmbeddingBatch:
        if not 1 <= limit <= 100 or after_id < 0:
            raise ValueError("Use batch size 1..100 and a nonnegative cursor")
        with self.sessions() as session:
            ids = list(
                session.scalars(
                    select(ArticleRecord.id)
                    .where(
                        ArticleRecord.id > after_id,
                        ArticleRecord.cleaning_version.is_not(None),
                        ArticleRecord.processing_status != "rejected",
                    )
                    .order_by(ArticleRecord.id)
                    .limit(limit)
                )
            )
        result = EmbeddingBatch(processed=len(ids), last_id=ids[-1] if ids else after_id)
        for article_id in ids:
            # Separate from ingestion's SQL commit. Lock per article so two indexers
            # cannot overwrite a newer embedding with an older snapshot.
            with self.sessions.begin() as session:
                record = session.scalar(
                    select(ArticleRecord).where(ArticleRecord.id == article_id).with_for_update()
                )
                if not eligible(record):
                    continue
                state = session.get(ArticleEmbeddingRecord, (article_id, self.store.collection))
                if state is None:
                    state = ArticleEmbeddingRecord(
                        article_id=article_id,
                        collection=self.store.collection,
                        model_version=self.provider.version,
                        dimensions=self.store.dimensions,
                        status="error",
                        attempted_at=datetime.now(UTC),
                    )
                    session.add(state)
                try:
                    input_ = article_input(record)
                    payload = {
                        "article_id": record.id,
                        "source": record.source,
                        "published_at": record.published_at.isoformat()
                        if record.published_at
                        else None,
                        "language": record.language,
                        "category": record.category,
                        "topics": sorted(topic.topic for topic in record.article_topics),
                        "entities": sorted({entity.canonical_name for entity in record.entities}),
                        "model_version": self.provider.version,
                        "input_hash": input_.input_hash,
                        "input_fields": input_.fields,
                        "text_truncated": input_.truncated,
                    }
                    payload_hash = digest(payload)
                    point = self.store.get(article_id)
                    same_vector = (
                        point is not None
                        and (point.payload or {}).get("input_hash") == input_.input_hash
                        and (point.payload or {}).get("model_version") == self.provider.version
                    )
                    if (
                        not refresh
                        and same_vector
                        and point.payload == payload
                        and (
                            state.status == "indexed"
                            and state.input_hash == input_.input_hash
                            and state.payload_hash == payload_hash
                        )
                    ):
                        result.unchanged += 1
                        continue
                    vector = normalized(
                        point.vector
                        if same_vector and not refresh
                        else self.provider.embed(input_.text),
                        self.store.dimensions,
                    )
                    self.store.upsert(article_id, vector, payload)
                    # A crash before SQL commit leaves an idempotent point upsert to replay.
                    state.status, state.input_hash, state.payload_hash = (
                        "indexed",
                        input_.input_hash,
                        payload_hash,
                    )
                    state.indexed_at = datetime.now(UTC)
                    state.error_type = None
                    result.indexed += 1
                except Exception as exc:
                    # Keep earlier successful hashes/timestamp so failed refreshes are visible.
                    state.status, state.error_type = "error", type(exc).__name__[:128]
                    logger.warning(
                        "embedding_article_failed article_id=%d error=%s",
                        article_id,
                        state.error_type,
                    )
                    result.errors += 1
                state.attempted_at = datetime.now(UTC)
        return result

    def prune(self) -> int:
        """Remove missing/rejected SQL articles only from this model's collection."""
        removed = 0
        for article_id in self.store.ids():
            with self.sessions.begin() as session:
                record = session.scalar(
                    select(ArticleRecord).where(ArticleRecord.id == article_id).with_for_update()
                )
                if not eligible(record):
                    self.store.delete(article_id)
                    session.execute(
                        delete(ArticleEmbeddingRecord).where(
                            ArticleEmbeddingRecord.article_id == article_id,
                            ArticleEmbeddingRecord.collection == self.store.collection,
                        )
                    )
                    removed += 1
        return removed

    @monitored("embedding_sync")
    def sync(self, batch_size: int = 50, *, refresh: bool = False) -> dict:
        totals = dict(processed=0, indexed=0, unchanged=0, errors=0, removed=0)
        cursor = 0
        while True:
            batch = self.backfill_batch(batch_size, after_id=cursor, refresh=refresh)
            if not batch.processed:
                break
            for key in totals:
                totals[key] += asdict(batch)[key]
            cursor = batch.last_id
            logger.info(
                "embedding_batch processed=%d indexed=%d errors=%d",
                batch.processed,
                batch.indexed,
                batch.errors,
            )
        totals["removed"] = self.prune()
        return {"collection": self.store.collection, **totals}
