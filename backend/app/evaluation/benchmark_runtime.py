import time
from datetime import datetime
from importlib.metadata import version

from qdrant_client import QdrantClient

from backend.app.database.session import create_session_factory
from backend.app.evaluation.cli import evaluation_database
from backend.app.evaluation.retrieval_benchmark import evaluate_cases, select_cases
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import FusionConfig, SearchRequest
from backend.app.services.article_embeddings import ArticleEmbeddingService
from backend.app.services.hybrid_search import fuse_results
from backend.app.services.lexical_search import LexicalSearchService
from backend.app.services.reranking import rerank_results
from backend.app.services.semantic_search import SemanticSearchService
from nlp.embeddings.store import QdrantEmbeddingStore
from nlp.multilingual.runtime import provider_for, reranker_for


def run_benchmark(dataset, settings, *, k=10, case_ids=None, progress=None, stage=None):
    select_cases(dataset, case_ids)  # Fail before allocating storage or loading models.
    if not 1 <= k <= 50:
        raise ValueError("Use k in 1..50")
    started = time.perf_counter()
    depth = max(10, k)
    pool = max(settings.reranking_candidate_limit, depth)
    fusion = FusionConfig(
        candidate_limit=max(settings.hybrid_candidate_limit, pool),
        rrf_k=settings.hybrid_rrf_k,
        semantic_weight=settings.hybrid_semantic_weight,
        lexical_weight=settings.hybrid_lexical_weight,
    )
    if stage:
        stage("database_setup")
    # Never connect retrieval to the live corpus or its vector directory.
    with evaluation_database(settings) as engine:
        sessions = create_session_factory(engine)
        with sessions.begin() as session:
            for article in dataset.articles:
                values = article.model_dump()
                values["url"] = str(article.url)
                session.add(
                    ArticleRecord(
                        **values,
                        retrieved_at=dataset.captured_at,
                        cleaning_version="eval-snapshot-v1",
                        processing_status="clean",
                    )
                )
        if stage:
            stage("model_loading_and_indexing")
        provider, reranker = provider_for(settings), reranker_for(settings)
        client = QdrantClient(":memory:")
        try:
            store = QdrantEmbeddingStore(client, "phase27_evaluation", provider.dimensions)
            indexing = ArticleEmbeddingService(sessions, provider, store).sync()
            if indexing["errors"] or indexing["indexed"] != len(dataset.articles):
                raise RuntimeError("Frozen corpus indexing incomplete")
            setup_seconds = time.perf_counter() - started

            def retrieve(inputs):
                context = inputs["time_context"]
                as_of = datetime.fromisoformat(context["as_of"].replace("Z", "+00:00"))
                request = SearchRequest(
                    q=inputs["question"],
                    **inputs["filters"],
                    date_from=context["date_from"],
                    date_to=context["date_to"],
                    recency="off",
                    limit=fusion.candidate_limit,
                )
                with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
                    with conn.begin():
                        snapshot = create_session_factory(conn)
                        lexical = LexicalSearchService(snapshot, as_of=as_of).search(
                            request.model_copy(update={"method": "lexical"}),
                        )
                        semantic = SemanticSearchService(
                            snapshot,
                            provider,
                            store,
                            as_of=as_of,
                        ).search(request.model_copy(update={"method": "semantic"}))
                        hybrid = fuse_results(semantic, lexical, fusion, pool)
                        reranked = rerank_results(
                            hybrid,
                            request.model_copy(update={"limit": depth, "rerank": True}),
                            snapshot,
                            reranker,
                            pool,
                            as_of=as_of,
                        )
                        return {
                            name: response.model_copy(
                                update={
                                    "results": response.results[:depth],
                                    "count": len(response.results[:depth]),
                                }
                            )
                            for name, response in {
                                "lexical": lexical,
                                "semantic": semantic,
                                "hybrid": hybrid,
                                "hybrid_reranked": reranked,
                            }.items()
                        }

            if stage:
                stage("retrieval")
            report = evaluate_cases(dataset, retrieve, k=k, case_ids=case_ids, progress=progress)
            report.update(
                {
                    "protocol_version": "frozen-retrieval-v1",
                    "setup_seconds": round(setup_seconds, 3),
                    "total_seconds": round(time.perf_counter() - started, 3),
                    "configuration": {
                        "fusion": fusion.model_dump(),
                        "reranking_candidate_limit": pool,
                        "recency_decay_applied": False,
                    },
                    "provenance": {
                        "storage": "isolated PostgreSQL schema and in-memory Qdrant",
                        "embedding_model_version": provider.version,
                        "reranking_model_version": reranker.version,
                        "versions": {
                            name: version(name)
                            for name in (
                                "newslens-ai",
                                "sentence-transformers",
                                "transformers",
                                "torch",
                                "qdrant-client",
                                "SQLAlchemy",
                            )
                        },
                        "indexing": indexing,
                    },
                }
            )
            return report
        finally:
            client.close()
