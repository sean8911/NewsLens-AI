from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.app.models.article import ArticleRecord
from backend.app.models.entity import EntityRecord, article_entities
from nlp.entities.extractor import ENTITY_TYPES, EntityResult, canonical_name


class EntityRepository:
    def __init__(self, session: Session):
        self.session = session

    def replace(self, article_id: int, result: EntityResult) -> None:
        self.session.execute(
            delete(article_entities).where(article_entities.c.article_id == article_id)
        )
        # Stable lock order avoids deadlocks when concurrent articles share several entities.
        for entity in sorted(
            result.entities, key=lambda item: (item.entity_type, item.canonical_name)
        ):
            statement = (
                insert(EntityRecord)
                .values(
                    name=entity.name,
                    canonical_name=entity.canonical_name,
                    entity_type=entity.entity_type,
                )
                .on_conflict_do_nothing(constraint="uq_entities_type_name")
                .returning(EntityRecord.id)
            )
            entity_id = self.session.scalar(statement)
            if entity_id is None:
                entity_id = self.session.scalar(
                    select(EntityRecord.id).where(
                        EntityRecord.entity_type == entity.entity_type,
                        EntityRecord.canonical_name == entity.canonical_name,
                    )
                )
            self.session.execute(
                insert(article_entities)
                .values(article_id=article_id, entity_id=entity_id)
                .on_conflict_do_nothing()
            )

    def search_articles(
        self, name: str, *, entity_type: str | None = None, limit: int = 20, offset: int = 0
    ) -> list[ArticleRecord]:
        if not name.strip() or len(name) > 512 or not 1 <= limit <= 100 or offset < 0:
            raise ValueError(
                "Use an entity name of 1..512 characters, limit 1..100 and offset >= 0"
            )
        if entity_type is not None and entity_type not in ENTITY_TYPES:
            raise ValueError("Unknown entity type")
        types = [entity_type] if entity_type else ENTITY_TYPES
        from sqlalchemy import or_

        matches = (
            select(article_entities.c.article_id)
            .join(EntityRecord)
            .where(
                or_(
                    *(
                        (EntityRecord.entity_type == kind)
                        & (EntityRecord.canonical_name == canonical_name(name, kind))
                        for kind in types
                    )
                )
            )
        )
        return list(
            self.session.scalars(
                select(ArticleRecord)
                .where(ArticleRecord.id.in_(matches), ArticleRecord.processing_status != "rejected")
                .order_by(ArticleRecord.retrieved_at.desc(), ArticleRecord.id.desc())
                .limit(limit)
                .offset(offset)
            )
        )
