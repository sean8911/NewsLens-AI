from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.app.models.article import ArticleRecord
from nlp.classification.classifier import HYPOTHESES, TopicClassifier
from nlp.embeddings.service import digest
from nlp.entities.extractor import EntityExtractor
from nlp.trending.scoring import TrendArticle, TrendEntity, TrendingScorer


class TrendingCapacityError(RuntimeError):
    """The complete comparison window exceeds the configured read bound."""


class TrendingService:
    def __init__(self, sessions, *, max_articles=10_000):
        if max_articles < 1:
            raise ValueError("Trend capacity must be positive")
        self.sessions, self.max_articles = sessions, max_articles

    def read(self, request):
        as_of = request.as_of or datetime.now(UTC)
        resolved = request.model_copy(update={"as_of": as_of})
        start = as_of - timedelta(hours=2 * request.window_hours)
        classifier, extractor = TopicClassifier(), EntityExtractor()
        excluded, inputs, no_entities = Counter(), [], 0
        with self.sessions() as session:
            records = list(
                session.scalars(
                    select(ArticleRecord)
                    .where(ArticleRecord.published_at >= start, ArticleRecord.published_at < as_of)
                    .options(
                        selectinload(ArticleRecord.article_topics),
                        selectinload(ArticleRecord.entities),
                    )
                    .order_by(ArticleRecord.id)
                    .limit(self.max_articles + 1)
                )
            )
            if len(records) > self.max_articles:
                raise TrendingCapacityError("Trend window exceeds configured article capacity")
            for article in records:
                if not article.cleaning_version or article.processing_status == "rejected":
                    excluded["unprocessed_or_rejected"] += 1
                    continue
                _, _, topic_hash = classifier.prepare(
                    article.title, article.description, article.content, article.language
                )
                if article.topic_status == "other" and article.topic_input_hash == topic_hash:
                    excluded["other_topic"] += 1
                    continue
                if article.topic_status != "classified" or article.topic_input_hash != topic_hash:
                    excluded["unavailable_or_stale_topics"] += 1
                    continue
                topics = tuple(
                    sorted(
                        {
                            label.topic
                            for label in article.article_topics
                            if label.topic in HYPOTHESES and label.topic != "Other"
                        }
                    )
                )
                if not topics:
                    excluded["no_supported_topic"] += 1
                    continue
                _, _, entity_hash, _ = extractor.prepare(
                    article.title, article.description, article.content, article.language
                )
                entities = (
                    tuple(
                        TrendEntity(e.name, e.canonical_name, e.entity_type)
                        for e in article.entities
                    )
                    if (
                        article.entity_status == "extracted"
                        and article.entity_input_hash == entity_hash
                    )
                    else ()
                )
                no_entities += not entities
                inputs.append(
                    TrendArticle(
                        article.id,
                        article.title,
                        article.source,
                        article.url,
                        article.published_at,
                        topics,
                        entities,
                    )
                )
        result = TrendingScorer().score(inputs, resolved)
        result.articles_scanned = len(records)
        result.excluded = dict(sorted(excluded.items()))
        result.articles_without_current_entities = no_entities
        result.input_hash = digest([result.input_hash, len(records), result.excluded, no_entities])
        return result
