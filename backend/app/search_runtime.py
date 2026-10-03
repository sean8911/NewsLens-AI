from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime

from backend.app.core.config import Settings
from backend.app.core.performance import monitored, stage
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.schemas.search import FusionConfig, SearchRequest, SearchResponse
from backend.app.schemas.temporal import RecencyConfig, TimeIntent
from backend.app.services.hybrid_search import HybridSearchService
from backend.app.services.lexical_search import LexicalSearchService
from backend.app.services.recency import apply_recency
from backend.app.services.reranking import rerank_results
from backend.app.services.semantic_search import SemanticSearchService
from backend.app.services.time_intent import resolve_time_intent
from nlp.embeddings.store import open_store
from nlp.multilingual.language import detect_language
from nlp.multilingual.runtime import provider_for, reranker_for


@contextmanager
def search_snapshot(
    request: SearchRequest,
    settings: Settings | None = None,
    *,
    as_of: datetime | None = None,
    time_query: str | None = None,
    frozen_intent: TimeIntent | None = None,
):
    """Yield results and SQL sessions from one snapshot; always release owned handles."""
    with ExitStack() as timing:
        timing.enter_context(stage("search"))
        settings = settings or Settings()
        temporal_request = (
            request if time_query is None else request.model_copy(update={"q": time_query})
        )
        if frozen_intent is not None:
            if (
                time_query is not None
                or (as_of is not None and as_of != frozen_intent.as_of)
                or request.date_from != frozen_intent.date_from
                or request.date_to != frozen_intent.date_to
            ):
                raise ValueError("Frozen retrieval intent must match the requested scope")
            intent = frozen_intent.model_copy(update={"retrieval_query": request.q})
        else:
            intent = resolve_time_intent(temporal_request, as_of or datetime.now(UTC))
        if time_query is not None:
            # Resolve dates from the current user question, never from old topic/title annotations.
            if not request.q.startswith(time_query):
                raise ValueError("Contextual query must start with the current question")
            intent = intent.model_copy(
                update={"retrieval_query": intent.retrieval_query + request.q[len(time_query) :]}
            )
        effective_request = request.model_copy(
            update={
                "q": intent.retrieval_query,
                "date_from": intent.date_from,
                "date_to": intent.date_to,
                "recency": "off",
            }
        )
        recency_config = RecencyConfig(
            decay=settings.recency_decay,
            half_life_days=settings.recency_half_life_days,
            strength=settings.recency_strength,
            reranking_weight=settings.recency_reranking_weight,
        )
        apply_time = intent.kind == "latest" and recency_config.strength > 0
        engine = create_database_engine(settings)
        store = None
        try:
            provider = None
            if request.method != "lexical":
                provider = provider_for(settings)
                store = open_store(settings, provider, create=False)
            # Retrieval, fusion and reranking hydration all see the same SQL snapshot.
            with engine.connect().execution_options(
                isolation_level="REPEATABLE READ"
            ) as connection:
                with connection.begin():
                    sessions = create_session_factory(connection)
                    candidate_limit = max(settings.reranking_candidate_limit, request.limit)
                    retrieval_request = effective_request.model_copy(
                        update={
                            "limit": candidate_limit
                            if request.rerank or apply_time
                            else request.limit,
                            "rerank": False,
                        }
                    )
                    if request.method == "lexical":
                        service = LexicalSearchService(sessions)
                    elif request.method == "semantic":
                        service = SemanticSearchService(sessions, provider, store)
                    else:
                        service = HybridSearchService(
                            SemanticSearchService(sessions, provider, store),
                            LexicalSearchService(sessions),
                            FusionConfig(
                                candidate_limit=settings.hybrid_candidate_limit,
                                rrf_k=settings.hybrid_rrf_k,
                                semantic_weight=settings.hybrid_semantic_weight,
                                lexical_weight=settings.hybrid_lexical_weight,
                            ),
                        )
                    response = service.search(retrieval_request)
                    if request.rerank:
                        response = rerank_results(
                            response,
                            effective_request.model_copy(
                                update={"limit": candidate_limit if apply_time else request.limit}
                            ),
                            sessions,
                            reranker_for(settings),
                            candidate_limit,
                        )
                    response = apply_recency(response, intent, recency_config, request.limit)
                    detected = detect_language(request.q)
                    response = response.model_copy(
                        update={
                            "query": request.q,
                            "query_language": detected.code,
                            "language_detection_fallback": detected.fallback,
                        }
                    )
                    timing.close()  # End search latency before evidence hydration by the caller.
                    yield response, sessions
        finally:
            if store is not None:
                store.client.close()
            engine.dispose()


@monitored("search")
def run_search(request: SearchRequest) -> SearchResponse:
    with search_snapshot(request) as (response, _):
        return response
