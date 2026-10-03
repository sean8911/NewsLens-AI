from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class SummaryFact(BaseModel):
    text: str
    citation_ids: list[int]


class EventSummaryCitation(BaseModel):
    citation_id: int
    article_id: int
    source: str
    title: str
    url: HttpUrl
    published_at: datetime | None
    field: Literal["title", "description", "content"]
    start: int
    end: int
    excerpt: str


class SummaryEntity(BaseModel):
    name: str
    citation_ids: list[int]


class EventSummaryDocument(BaseModel):
    what_happened: SummaryFact
    key_facts: list[SummaryFact]
    latest_update: SummaryFact | None = None
    latest_reported_at: datetime | None = None
    organizations: list[SummaryEntity] = Field(default_factory=list)
    people: list[SummaryEntity] = Field(default_factory=list)
    citations: list[EventSummaryCitation]
    source_count: int
    article_count: int
    total_event_articles: int
    omitted_article_ids: list[int]


class EventSummaryView(BaseModel):
    status: str = "not_attempted"
    document: EventSummaryDocument | None = None
    generated_at: datetime | None = None
    policy_version: str | None = None
    input_hash: str | None = None
    stale: bool = False
