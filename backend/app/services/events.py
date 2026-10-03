from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select

from backend.app.models.article import ArticleRecord
from backend.app.models.entity import EntityRecord, article_entities
from backend.app.models.event import ArticleEventRecord, EventClusterRun, EventRecord
from backend.app.schemas.event import (
    EventDetail,
    EventEntity,
    EventListRequest,
    EventListResponse,
    EventMember,
)
from backend.app.schemas.event_comparison import EventComparisonView
from backend.app.schemas.event_timeline import EventTimelineView, TimelineRequest
from backend.app.services.event_clustering import event_input_hash
from backend.app.services.event_summaries import membership_stale, prepare_summary, summary_view
from nlp.embeddings.service import digest
from nlp.events.comparison import EventComparator
from nlp.events.summarization import render_summary
from nlp.events.timeline import EventTimelineBuilder


class EventReadService:
    def __init__(self, sessions):
        self.sessions = sessions

    def _detail(self, session, event):
        rows = session.execute(
            select(ArticleEventRecord, ArticleRecord)
            .join(ArticleRecord, ArticleRecord.id == ArticleEventRecord.article_id)
            .where(ArticleEventRecord.event_id == event.id)
            .order_by(ArticleRecord.id)
        ).all()
        members = [
            EventMember(
                article_id=article.id,
                title=article.title,
                source=article.source,
                url=article.url,
                published_at=article.published_at,
                similarity_score=link.similarity_score,
                input_hash=link.input_hash,
                evidence=link.evidence,
                stale=(
                    link.input_hash != event_input_hash(article)
                    or not article.cleaning_version
                    or article.processing_status == "rejected"
                ),
            )
            for link, article in rows
        ]
        # Only current, visible member articles contribute entity associations.
        current_ids = [member.article_id for member in members if not member.stale]
        entities = {}
        if current_ids:
            entity_rows = session.execute(
                select(EntityRecord, article_entities.c.article_id)
                .join(article_entities, article_entities.c.entity_id == EntityRecord.id)
                .where(article_entities.c.article_id.in_(current_ids))
                .order_by(
                    EntityRecord.entity_type,
                    EntityRecord.canonical_name,
                    EntityRecord.id,
                    article_entities.c.article_id,
                )
            )
            for entity, article_id in entity_rows:
                if entity.id not in entities:
                    entities[entity.id] = EventEntity(
                        id=entity.id,
                        name=entity.name,
                        entity_type=entity.entity_type,
                        article_ids=[],
                    )
                entities[entity.id].article_ids.append(article_id)
        view = summary_view(session, event, rows)
        values = {
            column.name: getattr(event, column.name) for column in EventRecord.__table__.columns
        }
        values["summary"] = render_summary(view.document) if view.document else None
        return EventDetail(
            **values,
            event_summary=view,
            articles=members,
            related_entities=list(entities.values()),
            article_count=len(members),
            source_count=len({member.source for member in members}),
            stale=membership_stale(event, rows),
        )

    def list(self, request: EventListRequest) -> EventListResponse:
        with self.sessions() as session:
            ids = (
                select(ArticleEventRecord.event_id)
                .group_by(ArticleEventRecord.event_id)
                .having(func.count() >= request.min_articles)
            )
            total = session.scalar(
                select(func.count()).select_from(EventRecord).where(EventRecord.id.in_(ids))
            )
            events = session.scalars(
                select(EventRecord)
                .where(EventRecord.id.in_(ids))
                .order_by(EventRecord.last_updated_at.desc(), EventRecord.id.desc())
                .limit(request.limit)
                .offset(request.offset)
            )
            results = [self._detail(session, event) for event in events]
            return EventListResponse(count=len(results), total=total, results=results)

    def get(self, event_id: int) -> EventDetail | None:
        with self.sessions() as session:
            event = session.get(EventRecord, event_id)
            return self._detail(session, event) if event else None

    def comparison(self, event_id: int) -> EventComparisonView | None:
        with self.sessions() as session:
            event = session.get(EventRecord, event_id)
            if event is None:
                return None
            rows = session.execute(
                select(ArticleEventRecord, ArticleRecord)
                .join(ArticleRecord, ArticleRecord.id == ArticleEventRecord.article_id)
                .where(ArticleEventRecord.event_id == event_id)
                .order_by(ArticleRecord.id)
            ).all()
            sources, fingerprint = prepare_summary(event, rows)
            stale = membership_stale(event, rows)
            status, document = ("stale", None) if stale else EventComparator().compare(sources)
            return EventComparisonView(
                event_id=event_id,
                status=status,
                document=document,
                policy_version=EventComparator.version,
                input_hash=digest([fingerprint, EventComparator.version]),
                computed_at=datetime.now(UTC),
                stale=stale,
                notes=["Refresh event clustering before comparing changed membership."]
                if stale
                else [],
            )

    def timeline(self, event_id: int, request: TimelineRequest) -> EventTimelineView | None:
        with self.sessions() as session:
            event = session.get(EventRecord, event_id)
            if event is None:
                return None
            rows = session.execute(
                select(ArticleEventRecord, ArticleRecord)
                .join(ArticleRecord, ArticleRecord.id == ArticleEventRecord.article_id)
                .where(ArticleEventRecord.event_id == event_id)
                .order_by(ArticleRecord.id)
            ).all()
            sources, fingerprint = prepare_summary(event, rows)
            stale = membership_stale(event, rows)
            document = None if stale else EventTimelineBuilder().build(event_id, sources, request)
            return EventTimelineView(
                event_id=event_id,
                status="stale"
                if stale
                else "ready"
                if document.total_count
                else "insufficient_evidence",
                document=document,
                policy_version=EventTimelineBuilder.version,
                input_hash=digest([fingerprint, EventTimelineBuilder.version]),
                computed_at=datetime.now(UTC),
                stale=stale,
                notes=["Refresh event clustering before reading changed membership."]
                if stale
                else [],
            )

    def runs(self, limit: int = 5) -> list[dict]:
        if not 1 <= limit <= 100:
            raise ValueError("Use a limit from 1 to 100")
        with self.sessions() as session:
            return [
                {"id": run.id, "created_at": run.created_at.isoformat(), "report": run.report}
                for run in session.scalars(
                    select(EventClusterRun).order_by(EventClusterRun.id.desc()).limit(limit)
                )
            ]
