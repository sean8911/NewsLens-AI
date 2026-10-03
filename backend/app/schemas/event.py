from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.event_summary import EventSummaryView


class EventListRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    min_articles: int = Field(default=2, ge=1, le=1000)


class EventMember(BaseModel):
    article_id: int
    title: str
    source: str
    url: str
    published_at: datetime | None
    similarity_score: float
    input_hash: str
    evidence: dict
    stale: bool


class EventEntity(BaseModel):
    id: int
    name: str
    entity_type: str
    article_ids: list[int]


class EventDetail(BaseModel):
    related_entities: list[EventEntity] = Field(default_factory=list)
    event_summary: EventSummaryView = Field(default_factory=EventSummaryView)
    id: int
    title: str
    summary: str | None
    first_seen_at: datetime
    first_published_at: datetime | None
    last_published_at: datetime | None
    last_updated_at: datetime
    created_at: datetime
    representative_article_id: int | None
    policy_version: str
    model_version: str
    article_count: int
    source_count: int
    stale: bool
    articles: list[EventMember]


class EventListResponse(BaseModel):
    count: int
    total: int
    results: list[EventDetail]
