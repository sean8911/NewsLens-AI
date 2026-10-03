from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from backend.app.schemas.event_summary import EventSummaryCitation


class TimelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class TimelineItem(BaseModel):
    id: str
    event_id: int
    source_article_id: int
    timestamp: AwareDatetime | None
    timestamp_basis: Literal["publication", "unknown"]
    description: str
    citation: EventSummaryCitation


class TimelineExclusion(BaseModel):
    article_id: int
    reason: Literal["unsupported_language", "insufficient_text"]


class TimelineDocument(BaseModel):
    items: list[TimelineItem]
    undated_items: list[TimelineItem]
    excluded_articles: list[TimelineExclusion]
    count: int
    total_count: int
    dated_count: int
    undated_count: int
    total_event_articles: int
    source_count: int
    limit: int
    offset: int
    next_offset: int | None
    scope: list[str]


class EventTimelineView(BaseModel):
    event_id: int
    status: Literal["ready", "insufficient_evidence", "stale"]
    policy_version: str
    input_hash: str
    computed_at: datetime
    document: TimelineDocument | None = None
    stale: bool = False
    notes: list[str] = Field(default_factory=list)
