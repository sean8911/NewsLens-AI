from datetime import UTC, datetime

from sqlalchemy import delete, select, text

from backend.app.models.article import ArticleRecord
from backend.app.models.article_embedding import ArticleEmbeddingRecord
from backend.app.models.event import ArticleEventRecord, EventClusterRun, EventRecord
from backend.app.services.article_embeddings import article_input
from nlp.embeddings.service import digest, normalized
from nlp.events.clustering import ClusterConfig, EventDocument, actions, cluster, compare


def event_input_hash(article: ArticleRecord) -> str:
    return digest(
        [
            article_input(article).input_hash,
            article.published_at.isoformat() if article.published_at else None,
            article.retrieved_at.isoformat(),
            article.language,
        ]
    )


def load_document(session, article, provider, store):
    ledger = session.get(ArticleEmbeddingRecord, (article.id, store.collection))
    input_hash = article_input(article).input_hash
    if (
        not ledger
        or ledger.status != "indexed"
        or ledger.model_version != provider.version
        or (ledger.dimensions != store.dimensions or ledger.input_hash != input_hash)
    ):
        return None, "stale_or_missing_ledger"
    point = store.get(article.id)
    if point is None:
        return None, "missing_vector"
    payload = point.payload or {}
    if (
        payload.get("article_id") != article.id
        or payload.get("input_hash") != input_hash
        or (payload.get("model_version") != provider.version)
    ):
        return None, "stale_vector"
    try:
        vector = normalized(point.vector, store.dimensions)
    except (TypeError, ValueError):
        return None, "invalid_vector"
    return EventDocument(
        article.id, article.title, article.published_at, vector, article.language
    ), None


def match_event_ids(groups: list[list[int]], previous: dict[int, int]) -> dict[int, int]:
    """Reuse IDs by greatest membership overlap; deterministic split/merge tie breaks."""
    candidates = []
    for index, ids in enumerate(groups):
        counts = {}
        for article_id in ids:
            if article_id in previous:
                old = previous[article_id]
                counts[old] = counts.get(old, 0) + 1
        candidates.extend((-count, old, index) for old, count in counts.items())
    assigned, used = {}, set()
    for _, old, index in sorted(candidates):
        if index not in assigned and old not in used:
            assigned[index] = old
            used.add(old)
    return assigned


class EventClusteringService:
    def __init__(self, sessions, provider, store, config: ClusterConfig | None = None):
        self.sessions, self.provider, self.store = sessions, provider, store
        self.config = config or ClusterConfig()

    def sync(self) -> dict:
        with self.sessions.begin() as session:
            # Serialize event writers in this schema; ingestion/indexing also lock article
            # rows so a decision cannot commit over concurrent edits to its source text.
            session.execute(
                text(
                    "SELECT pg_advisory_xact_lock("
                    "hashtextextended(current_schema() || '/events', 0))"
                )
            )
            articles = list(
                session.scalars(
                    select(ArticleRecord)
                    .where(
                        ArticleRecord.cleaning_version.is_not(None),
                        ArticleRecord.processing_status != "rejected",
                    )
                    .order_by(ArticleRecord.id)
                    .limit(self.config.max_articles + 1)
                    .with_for_update()
                )
            )
            if len(articles) > self.config.max_articles:
                raise ValueError("Event corpus limit exceeded; increase limit or partition corpus")
            docs, excluded = {}, {}
            rows = {article.id: article for article in articles}
            for article in articles:
                document, reason = load_document(session, article, self.provider, self.store)
                if reason:
                    excluded[str(article.id)] = reason
                else:
                    docs[article.id] = document
            groups = cluster(list(docs.values()), self.config)
            previous_links = list(session.scalars(select(ArticleEventRecord)))
            previous = {link.article_id: link.event_id for link in previous_links}
            old_events = {event.id: event for event in session.scalars(select(EventRecord))}
            reused = match_event_ids(groups, previous)
            old_signatures = {
                event_id: sorted(
                    (link.article_id, link.input_hash)
                    for link in previous_links
                    if link.event_id == event_id
                )
                for event_id in old_events
            }
            # Replacing memberships and retiring empty/merged events is one transaction.
            session.execute(delete(ArticleEventRecord))
            assignments, retained, changed = {}, set(), 0
            now = datetime.now(UTC)
            for index, ids in enumerate(groups):
                event = old_events.get(reused.get(index))
                representative = event.representative_article_id if event else None
                if representative not in ids:
                    representative = min(
                        ids,
                        key=lambda id_: (
                            rows[id_].published_at or datetime.max.replace(tzinfo=UTC),
                            id_,
                        ),
                    )
                dated = [rows[id_].published_at for id_ in ids if rows[id_].published_at]
                values = dict(
                    title=rows[representative].title,
                    representative_article_id=representative,
                    first_seen_at=min(rows[id_].retrieved_at for id_ in ids),
                    first_published_at=min(dated) if dated else None,
                    last_published_at=max(dated) if dated else None,
                    policy_version=self.config.version,
                    model_version=self.provider.version,
                )
                signatures = sorted((id_, event_input_hash(rows[id_])) for id_ in ids)
                if event is None:
                    event = EventRecord(**values, last_updated_at=now)
                    session.add(event)
                    session.flush()
                    changed += 1
                elif old_signatures[event.id] != signatures or any(
                    getattr(event, name) != value for name, value in values.items()
                ):
                    for name, value in values.items():
                        setattr(event, name, value)
                    event.last_updated_at = now
                    event.summary = None
                    changed += 1
                retained.add(event.id)
                for id_ in ids:
                    pairs = [
                        compare(docs[id_], docs[other], self.config)
                        for other in ids
                        if other != id_
                    ]
                    assert all(pair["compatible"] for pair in pairs)
                    weakest = min(pairs, key=lambda pair: pair["cosine_similarity"], default=None)
                    session.add(
                        ArticleEventRecord(
                            article_id=id_,
                            event_id=event.id,
                            similarity_score=weakest["cosine_similarity"] if weakest else 1.0,
                            input_hash=event_input_hash(rows[id_]),
                            evidence={
                                "rule": "complete_link",
                                "compared_members": len(pairs),
                                "weakest_pair": weakest,
                                "actions": actions(docs[id_]),
                            },
                        )
                    )
                    assignments[str(id_)] = event.id
            removed = sorted(old_events.keys() - retained)
            if removed:
                session.execute(delete(EventRecord).where(EventRecord.id.in_(removed)))
            report = {
                "articles_checked": len(articles),
                "articles_clustered": len(docs),
                "events": len(groups),
                "multi_article_events": sum(len(ids) > 1 for ids in groups),
                "events_changed": changed,
                "retired_event_ids": removed,
                "excluded": excluded,
                "policy": self.config.model_dump(),
                "policy_version": self.config.version,
                "model_version": self.provider.version,
                "collection": self.store.collection,
                "previous_assignments": {str(id_): event for id_, event in previous.items()},
                "assignments": assignments,
            }
            session.add(EventClusterRun(report=report))
            return report

    def compare_articles(self, article_a: int, article_b: int) -> dict:
        if article_a <= 0 or article_b <= 0 or article_a == article_b:
            raise ValueError("Use two distinct positive article IDs")
        with self.sessions() as session:
            docs = []
            for id_ in (article_a, article_b):
                article = session.get(ArticleRecord, id_)
                if (
                    article is None
                    or not article.cleaning_version
                    or (article.processing_status == "rejected")
                ):
                    raise ValueError("Article unavailable for clustering")
                doc, reason = load_document(session, article, self.provider, self.store)
                if reason:
                    return {"article_id": id_, "available": False, "reason": reason}
                docs.append(doc)
            return {
                "available": True,
                "policy": self.config.model_dump(),
                **compare(*docs, self.config),
            }
