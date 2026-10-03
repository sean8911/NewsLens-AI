from datetime import date, datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.stored_article import StoredEntity, StoredKeyword, StoredTopic

MAX_ID = 2**63 - 1
T = TypeVar("T")


class Pagination(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=1_000_000)


class ArticleListRequest(Pagination):
    source: str | None = Field(default=None, min_length=1, max_length=200)
    category: str | None = Field(default=None, min_length=1, max_length=200)
    language: str | None = Field(default=None, pattern=r"^[a-z]{2,16}$")
    topic: str | None = Field(default=None, min_length=1, max_length=48)
    entity_id: int | None = Field(default=None, gt=0, le=MAX_ID)
    date_from: date | None = Field(default=None, description="Inclusive UTC publication date")
    date_to: date | None = Field(default=None, description="Inclusive UTC publication date")

    @model_validator(mode="after")
    def ordered_dates(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self


class EntityListRequest(Pagination):
    name: str | None = Field(default=None, min_length=1, max_length=512)
    entity_type: Literal["PERSON", "ORGANIZATION", "LOCATION", "COUNTRY", "PRODUCT"] | None = None


class Page(BaseModel, Generic[T]):
    results: list[T]
    count: int = Field(ge=0)
    total: int = Field(ge=0)
    limit: int
    offset: int
    has_more: bool


class ArticleCard(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    description: str | None
    url: str
    source: str
    author: str | None
    published_at: datetime | None
    retrieved_at: datetime
    category: str
    language: str | None
    summary: str | None
    article_topics: list[StoredTopic]
    entities: list[StoredEntity]


class ArticleDetail(ArticleCard):
    content: str | None
    content_available: bool
    rss_categories: list[str]
    keywords: list[StoredKeyword]


class TopicItem(BaseModel):
    topic: str
    article_count: int = Field(ge=0)


class EntityItem(StoredEntity):
    article_count: int = Field(ge=0)


class EntityDetail(BaseModel):
    entity: EntityItem
    articles: Page[ArticleCard]


class ErrorResponse(BaseModel):
    detail: str


class ValidationIssue(BaseModel):
    type: str


class ValidationErrorResponse(ErrorResponse):
    errors: list[ValidationIssue]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["NewsLens AI"] = "NewsLens AI"
    version: str


class SystemStatus(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    checked_at: datetime
    database: Literal["available", "unavailable", "unconfigured"]
    article_count: int | None = None
    last_successful_news_update: datetime | None = None
    last_successful_update_status: Literal["succeeded", "partial"] | None = None
    latest_article_published_at: datetime | None = None
    latest_article_retrieved_at: datetime | None = None
    last_update_attempt_at: datetime | None = None
    last_update_attempt_status: (
        Literal["running", "succeeded", "partial", "failed", "interrupted"] | None
    ) = None
    # Do not claim inference readiness merely because a model folder exists.
    model_runtime: Literal["not_checked"] = "not_checked"
    vector_runtime: Literal["not_checked"] = "not_checked"
