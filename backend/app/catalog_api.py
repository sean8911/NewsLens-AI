import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Response

from backend.app.core.version import VERSION
from backend.app.schemas.catalog import (
    MAX_ID,
    ArticleCard,
    ArticleDetail,
    ArticleListRequest,
    EntityDetail,
    EntityItem,
    EntityListRequest,
    ErrorResponse,
    HealthResponse,
    Page,
    Pagination,
    SystemStatus,
    TopicItem,
)

logger = logging.getLogger(__name__)
ERRORS = {503: {"model": ErrorResponse, "description": "Required storage unavailable"}}
NOT_FOUND = {**ERRORS, 404: {"model": ErrorResponse, "description": "Resource not found"}}


def read(call, *args):
    try:
        return call(*args)
    except Exception as exc:
        logger.error("catalog_read_failed error=%s", type(exc).__name__)
        raise HTTPException(503, "News data unavailable. Try again later.") from None


def create_catalog_router(catalog, system_status):
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse, tags=["System"])
    def health():
        """Process liveness; does not connect to storage or load models."""
        return HealthResponse(version=VERSION)

    @router.get(
        "/system/status",
        response_model=SystemStatus,
        tags=["System"],
        responses={503: {"model": SystemStatus, "description": "Database unavailable"}},
    )
    def status(response: Response):
        """Database read status and news-update freshness; model/vector inference is not probed."""
        result = system_status()
        if result.status == "degraded":
            response.status_code = 503
        return result

    @router.get("/articles", response_model=Page[ArticleCard], tags=["Articles"], responses=ERRORS)
    def articles(request: Annotated[ArticleListRequest, Query()]):
        """Browse cleaned articles by publication date (newest first, undated last)."""
        return read(catalog.articles, request)

    @router.get(
        "/articles/{article_id}",
        response_model=ArticleDetail,
        tags=["Articles"],
        responses=NOT_FOUND,
    )
    def article(article_id: Annotated[int, Path(gt=0, le=MAX_ID)]):
        """Read one cleaned article, its stored body, topics, entities and keywords."""
        result = read(catalog.article, article_id)
        if result is None:
            raise HTTPException(404, "Article not found")
        return result

    @router.get("/topics", response_model=Page[TopicItem], tags=["Topics"], responses=ERRORS)
    def topics(request: Annotated[Pagination, Query()]):
        """List assigned topics and visible article counts, ordered by count then name."""
        return read(catalog.topics, request)

    @router.get("/entities", response_model=Page[EntityItem], tags=["Entities"], responses=ERRORS)
    def entities(request: Annotated[EntityListRequest, Query()]):
        """Discover entities with visible articles; name matches a literal substring."""
        return read(catalog.entities, request)

    @router.get(
        "/entities/{entity_id}", response_model=EntityDetail, tags=["Entities"], responses=NOT_FOUND
    )
    def entity(
        entity_id: Annotated[int, Path(gt=0, le=MAX_ID)], request: Annotated[Pagination, Query()]
    ):
        """Read an entity and a paginated list of its visible articles."""
        result = read(catalog.entity, entity_id, request)
        if result is None:
            raise HTTPException(404, "Entity not found")
        return result

    return router
