import math

from qdrant_client import models
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.performance import measured
from backend.app.models.article import ArticleRecord
from backend.app.models.article_embedding import ArticleEmbeddingRecord
from backend.app.schemas.search import SearchHit, SearchRequest, SearchResponse
from backend.app.services.article_embeddings import article_input, current_summary
from backend.app.services.search_filters import apply_article_filters
from nlp.embeddings.service import EmbeddingService, normalized
from nlp.embeddings.store import QdrantEmbeddingStore


class SemanticSearchService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        provider: EmbeddingService,
        store: QdrantEmbeddingStore,
        *,
        as_of=None,
    ):
        self.sessions, self.provider, self.store = sessions, provider, store
        self.as_of = as_of
        if provider.dimensions != store.dimensions:
            raise ValueError("Search and index dimensions differ")

    @measured("retrieval")
    def search(self, request: SearchRequest) -> SearchResponse:
        # Asymmetric models require query prefixes, while indexed articles use passage prefixes.
        embed_query = getattr(self.provider, "embed_query", self.provider.embed)
        vector = normalized(embed_query(request.q), self.store.dimensions)
        with self.sessions() as session:
            query = (
                select(ArticleRecord, ArticleEmbeddingRecord)
                .join(ArticleEmbeddingRecord, ArticleEmbeddingRecord.article_id == ArticleRecord.id)
                .where(
                    ArticleEmbeddingRecord.collection == self.store.collection,
                    ArticleEmbeddingRecord.model_version == self.provider.version,
                    ArticleEmbeddingRecord.dimensions == self.store.dimensions,
                    ArticleEmbeddingRecord.status == "indexed",
                )
            )
            query = apply_article_filters(query, request, as_of=self.as_of)
            # SQL determines membership before ranking. This avoids relying on stale
            # Qdrant metadata to include/exclude articles after standalone NLP updates.
            candidates = {}
            for article, state in session.execute(query):
                current_hash = article_input(article).input_hash
                if state.input_hash == current_hash:
                    candidates[article.id] = (article, current_hash)
            results = []
            offset = 0
            while candidates and len(results) < request.limit:
                points = self.store.client.query_points(
                    collection_name=self.store.collection,
                    query=vector,
                    query_filter=models.Filter(
                        must=[models.HasIdCondition(has_id=list(candidates))]
                    ),
                    limit=50,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                ).points
                if not points:
                    break
                offset += len(points)
                for point in points:
                    candidate = candidates.get(point.id)
                    payload = point.payload or {}
                    if candidate is None:
                        continue
                    article, current_hash = candidate
                    if (
                        payload.get("article_id") != article.id
                        or payload.get("input_hash") != current_hash
                        or payload.get("model_version") != self.provider.version
                        or not math.isfinite(point.score)
                    ):
                        continue
                    results.append(
                        SearchHit(
                            article_id=article.id,
                            title=article.title,
                            source=article.source,
                            published_at=article.published_at,
                            relevance_score=point.score,
                            summary=current_summary(article),
                            url=article.url,
                            language=article.language,
                            category=article.category,
                        )
                    )
                    if len(results) == request.limit:
                        break
                if len(points) < 50:
                    break
        return SearchResponse(
            query=request.q,
            collection=self.store.collection,
            model_version=self.provider.version,
            count=len(results),
            results=results,
        )
