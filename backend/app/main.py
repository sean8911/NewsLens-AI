import logging
from collections.abc import Callable
from threading import Lock
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.responses import HTMLResponse

from backend.app.catalog_api import create_catalog_router
from backend.app.catalog_runtime import CatalogRuntime, read_system_status
from backend.app.chat_runtime import run_chat
from backend.app.conversation_runtime import create_conversation, read_conversation
from backend.app.core.api_security import APISecurityMiddleware, validation_error
from backend.app.core.config import Settings
from backend.app.core.version import VERSION
from backend.app.event_runtime import (
    compare_event_sources,
    get_event,
    get_event_timeline,
    list_events,
)
from backend.app.research_runtime import run_research
from backend.app.schemas.catalog import MAX_ID, ErrorResponse, ValidationErrorResponse
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.schemas.conversation import ConversationHistory, ConversationInfo
from backend.app.schemas.event import EventDetail, EventListRequest, EventListResponse
from backend.app.schemas.event_comparison import EventComparisonView
from backend.app.schemas.event_timeline import EventTimelineView, TimelineRequest
from backend.app.schemas.research import ResearchRequest, ResearchResponse
from backend.app.schemas.search import SearchRequest, SearchResponse
from backend.app.schemas.trending import TrendingRequest, TrendingResponse
from backend.app.search_runtime import run_search
from backend.app.services.conversation_store import ConversationError
from backend.app.services.trending import TrendingCapacityError
from backend.app.trending_runtime import read_trending
from nlp.embeddings.local_client import IndexBusyError

logger = logging.getLogger(__name__)


def create_app(
    search: Callable[[SearchRequest], SearchResponse] | None = None,
    chat: Callable[[ChatRequest], ChatResponse] | None = None,
    events: Callable[[EventListRequest], EventListResponse] | None = None,
    event_detail: Callable[[int], EventDetail | None] | None = None,
    event_comparison: Callable[[int], EventComparisonView | None] | None = None,
    event_timeline: Callable[[int, TimelineRequest], EventTimelineView | None] | None = None,
    trending: Callable[[TrendingRequest], TrendingResponse] | None = None,
    conversation_create: Callable | None = None,
    conversation_history: Callable | None = None,
    research: Callable[[ResearchRequest], ResearchResponse] | None = None,
    catalog: CatalogRuntime | None = None,
    system_status: Callable | None = None,
    security_settings: Settings | None = None,
) -> FastAPI:
    application = FastAPI(
        responses={
            408: {"model": ErrorResponse, "description": "Request body timeout"},
            413: {"model": ErrorResponse, "description": "Request body too large"},
            414: {"model": ErrorResponse, "description": "Query too large"},
            422: {"model": ValidationErrorResponse, "description": "Invalid request"},
            429: {"model": ErrorResponse, "description": "Rate limited; see Retry-After"},
            500: {"model": ErrorResponse, "description": "Internal service error"},
            503: {"model": ErrorResponse, "description": "Dependency unavailable or capacity busy"},
        },
        title="NewsLens AI",
        version=VERSION,
        description=(
            "Local news intelligence API. Browse cleaned articles, topics and entities; "
            "search news, explore events, and ask citation-grounded questions. "
            "Catalog lists use limit/offset pagination. Search returns ranked top-K hits."
        ),
    )
    application.add_middleware(APISecurityMiddleware, settings=security_settings or Settings())
    application.add_exception_handler(RequestValidationError, validation_error)

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def api_home():
        return (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>NewsLens API</title><body><h1>NewsLens API is running</h1>"
            "<p>This is the backend. Use the website to chat with your news.</p>"
            '<p><a href="http://127.0.0.1:3000/chat">Open NewsLens chat / 打开新闻对话</a></p>'
            "<p>If the website does not open, start the frontend terminal first.</p>"
            '<p><a href="/docs">API documentation</a></p></body></html>'
        )

    application.include_router(
        create_catalog_router(catalog or CatalogRuntime(), system_status or read_system_status)
    )
    research = research or run_research
    conversation_create = conversation_create or create_conversation
    conversation_history = conversation_history or read_conversation
    search = search or run_search
    chat = chat or run_chat
    events = events or list_events
    event_detail = event_detail or get_event
    event_comparison = event_comparison or compare_event_sources
    event_timeline = event_timeline or get_event_timeline
    trending = trending or read_trending
    lock = Lock()

    @application.post("/research", response_model=ResearchResponse)
    def research_articles(request: ResearchRequest) -> ResearchResponse:
        try:
            with lock:
                return research(request)
        except Exception as exc:
            logger.error("research_failed error=%s", type(exc).__name__)
            raise HTTPException(
                503, "Research unavailable. Check PostgreSQL, Qdrant and local models."
            ) from None

    @application.post("/conversations", response_model=ConversationInfo, status_code=201)
    def start_conversation():
        try:
            return conversation_create()
        except Exception as exc:
            logger.error("conversation_create_failed error=%s", type(exc).__name__)
            raise HTTPException(503, "Conversation storage unavailable.") from None

    @application.get(
        "/conversations/{conversation_id}",
        response_model=ConversationHistory,
        responses={404: {"model": ErrorResponse}},
    )
    def conversation_messages(
        conversation_id: UUID,
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
        before: Annotated[int | None, Query(ge=1)] = None,
    ):
        try:
            return conversation_history(conversation_id, limit, before)
        except ConversationError as exc:
            raise HTTPException(exc.status_code, str(exc)) from None
        except Exception as exc:
            logger.error("conversation_history_failed error=%s", type(exc).__name__)
            raise HTTPException(503, "Conversation storage unavailable.") from None

    @application.get(
        "/search",
        response_model=SearchResponse,
        responses={503: {"model": ErrorResponse, "description": "Search dependencies unavailable"}},
    )
    def search_articles(request: Annotated[SearchRequest, Query()]) -> SearchResponse:
        try:
            if request.method == "lexical":
                return search(request)
            # Local Qdrant cannot be opened twice concurrently. Each call creates and
            # closes its handles in this same worker thread, including SQLite handles.
            with lock:
                return search(request)
        except IndexBusyError:
            raise HTTPException(
                503, "The news index is busy. Retry shortly.", headers={"Retry-After": "5"}
            ) from None
        except Exception as exc:
            logger.error("%s_search_failed error=%s", request.method, type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Search unavailable. Check retrieval dependencies and local reranking model."
                    if request.rerank
                    else "Search unavailable. Check PostgreSQL and lexical search migration."
                    if request.method == "lexical"
                    else "Search unavailable. Check PostgreSQL, embedding model and Qdrant index."
                ),
            ) from None

    @application.post(
        "/search",
        response_model=SearchResponse,
        responses={503: {"model": ErrorResponse, "description": "Search dependencies unavailable"}},
    )
    def search_body(request: SearchRequest) -> SearchResponse:
        """Search with a JSON body; equivalent to the existing GET /search query interface."""
        return search_articles(request)

    @application.post(
        "/chat",
        response_model=ChatResponse,
        responses={
            503: {"model": ErrorResponse, "description": "Chat dependencies unavailable"},
            404: {"model": ErrorResponse, "description": "Conversation not found"},
            409: {"model": ErrorResponse, "description": "Conversation revision conflict"},
        },
    )
    def chat_articles(request: ChatRequest) -> ChatResponse:
        try:
            # Share the search lock: chat also opens the local Qdrant collection.
            with lock:
                return chat(request)
        except ConversationError as exc:
            raise HTTPException(exc.status_code, str(exc)) from None
        except IndexBusyError:
            raise HTTPException(
                503, "The news index is busy. Retry shortly.", headers={"Retry-After": "5"}
            ) from None
        except Exception as exc:
            logger.error("chat_failed error=%s", type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail="Chat unavailable. Check PostgreSQL, Qdrant, retrieval models and LLM.",
            ) from None

    @application.get("/events", response_model=EventListResponse)
    def read_events(request: Annotated[EventListRequest, Query()]) -> EventListResponse:
        try:
            return events(request)
        except Exception as exc:
            logger.error("events_read_failed error=%s", type(exc).__name__)
            raise HTTPException(
                503, "Events unavailable. Check PostgreSQL and event migration."
            ) from None

    @application.get(
        "/events/{event_id}", response_model=EventDetail, responses={404: {"model": ErrorResponse}}
    )
    def read_event(event_id: Annotated[int, Path(gt=0, le=MAX_ID)]) -> EventDetail:
        try:
            result = event_detail(event_id)
        except Exception as exc:
            logger.error("event_read_failed error=%s", type(exc).__name__)
            raise HTTPException(
                503, "Events unavailable. Check PostgreSQL and event migration."
            ) from None
        if result is None:
            raise HTTPException(404, "Event not found")
        return result

    @application.get(
        "/events/{event_id}/comparison",
        response_model=EventComparisonView,
        responses={
            404: {"model": ErrorResponse, "description": "Event not found"},
            503: {"model": ErrorResponse, "description": "Database unavailable"},
        },
    )
    def read_event_comparison(
        event_id: Annotated[int, Path(gt=0, le=MAX_ID)],
    ) -> EventComparisonView:
        try:
            result = event_comparison(event_id)
        except Exception as exc:
            logger.error("event_comparison_failed error=%s", type(exc).__name__)
            raise HTTPException(
                503, "Comparison unavailable. Check PostgreSQL and event migration."
            ) from None
        if result is None:
            raise HTTPException(404, "Event not found")
        return result

    @application.get(
        "/events/{event_id}/timeline",
        response_model=EventTimelineView,
        responses={
            404: {"model": ErrorResponse, "description": "Event not found"},
            503: {"model": ErrorResponse, "description": "Database unavailable"},
        },
    )
    def read_event_timeline(
        event_id: Annotated[int, Path(gt=0, le=MAX_ID)],
        request: Annotated[TimelineRequest, Query()],
    ) -> EventTimelineView:
        try:
            result = event_timeline(event_id, request)
        except Exception as exc:
            logger.error("event_timeline_failed error=%s", type(exc).__name__)
            raise HTTPException(
                503, "Timeline unavailable. Check PostgreSQL and event migration."
            ) from None
        if result is None:
            raise HTTPException(404, "Event not found")
        return result

    @application.get(
        "/trending",
        response_model=TrendingResponse,
        responses={
            503: {
                "model": ErrorResponse,
                "description": "Database unavailable or window exceeds capacity",
            }
        },
    )
    def trending_topics(request: Annotated[TrendingRequest, Query()]) -> TrendingResponse:
        try:
            return trending(request)
        except TrendingCapacityError:
            raise HTTPException(
                503, "Trend window exceeds capacity. Use a shorter window."
            ) from None
        except Exception as exc:
            logger.error("trending_read_failed error=%s", type(exc).__name__)
            raise HTTPException(
                503, "Trending unavailable. Check PostgreSQL and topic data."
            ) from None

    tags = {
        "search": "Search",
        "chat": "Chat",
        "conversations": "Conversations",
        "research": "Research",
        "events": "Events",
        "trending": "Trending",
    }
    for route in application.routes:
        if isinstance(route, APIRoute):
            if not route.tags:
                route.tags = [tags.get(route.path.split("/")[1], "News")]
    return application


# No model loads, downloads, database connections or background jobs at import time.
app = create_app()
