"""Read-only public article, topic and entity views over one SQL snapshot."""

from sqlalchemy import func, select

from backend.app.models.article import ArticleRecord
from backend.app.models.article_topic import ArticleTopicRecord
from backend.app.models.entity import EntityRecord, article_entities
from backend.app.schemas.catalog import (
    ArticleCard,
    ArticleDetail,
    ArticleListRequest,
    EntityDetail,
    EntityItem,
    Page,
    TopicItem,
)
from backend.app.services.article_embeddings import current_summary
from backend.app.services.search_filters import apply_article_filters


def page(results, total, request):
    return dict(
        results=results,
        count=len(results),
        total=total,
        limit=request.limit,
        offset=request.offset,
        has_more=request.offset + len(results) < total,
    )


def visible_ids():
    return apply_article_filters(select(ArticleRecord.id), ArticleListRequest())


def public_article(record, *, detail=False):
    schema = ArticleDetail if detail else ArticleCard
    result = schema.model_validate(record)
    # A stale persisted summary must not be presented as a summary of changed content.
    return result.model_copy(update={"summary": current_summary(record)})


class CatalogService:
    def __init__(self, session):
        self.session = session

    def articles(self, request):
        query = apply_article_filters(select(ArticleRecord), request)
        if request.topic is not None:
            query = query.where(
                ArticleRecord.article_topics.any(ArticleTopicRecord.topic == request.topic)
            )
        if request.entity_id is not None:
            query = query.where(ArticleRecord.entities.any(EntityRecord.id == request.entity_id))
        total = self.session.scalar(select(func.count()).select_from(query.subquery()))
        records = self.session.scalars(
            query.order_by(
                ArticleRecord.published_at.desc().nulls_last(),
                ArticleRecord.retrieved_at.desc(),
                ArticleRecord.id.desc(),
            )
            .offset(request.offset)
            .limit(request.limit)
        ).all()
        return Page[ArticleCard](**page([public_article(row) for row in records], total, request))

    def article(self, article_id):
        record = self.session.scalar(
            select(ArticleRecord).where(
                ArticleRecord.id == article_id, ArticleRecord.id.in_(visible_ids())
            )
        )
        return public_article(record, detail=True) if record else None

    def topics(self, request):
        query = (
            select(ArticleTopicRecord.topic, func.count().label("article_count"))
            .where(ArticleTopicRecord.article_id.in_(visible_ids()))
            .group_by(ArticleTopicRecord.topic)
        )
        total = self.session.scalar(select(func.count()).select_from(query.subquery()))
        rows = self.session.execute(
            query.order_by(func.count().desc(), ArticleTopicRecord.topic)
            .offset(request.offset)
            .limit(request.limit)
        ).all()
        return Page[TopicItem](
            **page(
                [TopicItem(topic=row.topic, article_count=row.article_count) for row in rows],
                total,
                request,
            )
        )

    def _entities(self):
        counts = (
            select(article_entities.c.entity_id, func.count().label("article_count"))
            .where(article_entities.c.article_id.in_(visible_ids()))
            .group_by(article_entities.c.entity_id)
            .subquery()
        )
        return select(EntityRecord, counts.c.article_count).join(
            counts, counts.c.entity_id == EntityRecord.id
        )

    def entities(self, request):
        query = self._entities()
        if request.name is not None:
            # Literal substring, so '%' and '_' in names never become SQL wildcards.
            query = query.where(
                func.lower(EntityRecord.name).contains(request.name.lower(), autoescape=True)
            )
        if request.entity_type is not None:
            query = query.where(EntityRecord.entity_type == request.entity_type)
        total = self.session.scalar(select(func.count()).select_from(query.subquery()))
        rows = self.session.execute(
            query.order_by(EntityRecord.canonical_name, EntityRecord.entity_type, EntityRecord.id)
            .offset(request.offset)
            .limit(request.limit)
        ).all()
        results = [
            EntityItem(
                **{
                    key: getattr(row, key)
                    for key in ("id", "name", "canonical_name", "entity_type")
                },
                article_count=count,
            )
            for row, count in rows
        ]
        return Page[EntityItem](**page(results, total, request))

    def entity(self, entity_id, request):
        row = self.session.execute(self._entities().where(EntityRecord.id == entity_id)).first()
        if row is None:
            return None
        entity, count = row
        return EntityDetail(
            entity=EntityItem(
                id=entity.id,
                name=entity.name,
                canonical_name=entity.canonical_name,
                entity_type=entity.entity_type,
                article_count=count,
            ),
            articles=self.articles(ArticleListRequest(entity_id=entity_id, **request.model_dump())),
        )
