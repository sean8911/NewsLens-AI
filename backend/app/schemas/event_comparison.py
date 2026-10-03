from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.schemas.event_summary import EventSummaryCitation, SummaryFact


class ReportDifference(BaseModel):
    kind: Literal["numbers", "wording"]
    explanation: str
    reports: list[SummaryFact]


class ReportingEmphasis(BaseModel):
    topic: str
    explanation: str
    reports: list[SummaryFact]
    other_compared_sources: list[str]


class EventComparisonDocument(BaseModel):
    agreement: list[SummaryFact]
    differences: list[ReportDifference]
    additional_details: list[SummaryFact]
    emphasis: list[ReportingEmphasis]
    citations: list[EventSummaryCitation]
    source_count: int
    article_count: int
    total_event_articles: int
    omitted_article_ids: list[int]
    scope: list[str]


class EventComparisonView(BaseModel):
    event_id: int
    status: Literal[
        "compared", "insufficient_sources", "insufficient_text", "unsupported_language", "stale"
    ]
    policy_version: str
    input_hash: str
    computed_at: datetime
    document: EventComparisonDocument | None = None
    stale: bool = False
    notes: list[str] = Field(default_factory=list)
