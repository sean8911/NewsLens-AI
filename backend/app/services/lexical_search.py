from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.performance import measured
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchHit, SearchRequest, SearchResponse
from backend.app.services.article_embeddings import current_summary
from backend.app.services.search_filters import apply_article_filters


class LexicalSearchService:
    def __init__(self, sessions: sessionmaker[Session], *, as_of=None):
        self.sessions = sessions
        self.as_of = as_of

    @measured("retrieval")
    def search(self, request: SearchRequest) -> SearchResponse:
        vector = func.newslens_lexical_vector(
            ArticleRecord.title, ArticleRecord.description, ArticleRecord.content
        )
        tsquery = func.websearch_to_tsquery("pg_catalog.simple", request.q)
        # PostgreSQL cover-density rank, equal field weights, rank/(rank+1) normalization.
        # These scores are separate from cosine similarity; no score fusion occurs here.
        score = func.ts_rank_cd(vector, tsquery, 32).label("lexical_rank")
        statement = (
            apply_article_filters(select(ArticleRecord, score), request, as_of=self.as_of)
            .where(
                vector.op("@@")(tsquery),
                func.querytree(tsquery).not_in(["", "T"]),  # Require a positive searchable term.
            )
            .order_by(score.desc(), ArticleRecord.id.asc())
            .limit(request.limit)
        )
        with self.sessions() as session:
            results = [
                SearchHit(
                    article_id=article.id,
                    title=article.title,
                    source=article.source,
                    published_at=article.published_at,
                    relevance_score=float(rank),
                    summary=current_summary(article),
                    url=article.url,
                    language=article.language,
                    category=article.category,
                )
                for article, rank in session.execute(statement)
            ]
        return SearchResponse(
            query=request.q, method="lexical", count=len(results), results=results
        )
