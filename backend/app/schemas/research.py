from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.schemas.chat import ChatSource
from backend.app.schemas.search import SearchRequest
from backend.app.schemas.temporal import TimeIntent


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    answer_language: Literal["en", "ms", "zh"] | None = None
    recency: Literal["auto", "off", "latest"] = "auto"
    source: str | None = Field(default=None, min_length=1, max_length=200)
    language: str | None = Field(default=None, pattern=r"^[a-z]{2,16}$")
    category: str | None = Field(default=None, min_length=1, max_length=200)
    date_from: date | None = None
    date_to: date | None = None
    compare: bool = False
    compare_sources: list[str] = Field(default_factory=list, max_length=3)
    max_searches: int = Field(default=3, ge=1, le=4)
    max_articles: int = Field(default=8, ge=1, le=8)

    @field_validator("compare_sources")
    @classmethod
    def publisher_names(cls, values):
        values = [" ".join(value.split()) for value in values]
        if any(not 1 <= len(value) <= 200 for value in values):
            raise ValueError("Publisher names must contain 1 to 200 characters")
        if len({value.casefold() for value in values}) != len(values):
            raise ValueError("Publisher names must be distinct")
        return values

    @model_validator(mode="after")
    def valid_scope(self) -> Self:
        self.search_request()
        if self.source and any(value != self.source for value in self.compare_sources):
            raise ValueError("Comparison publishers cannot widen the source filter")
        return self

    def search_request(self) -> SearchRequest:
        return SearchRequest(
            q=self.question,
            method="hybrid",
            rerank=True,
            limit=5,
            **self.model_dump(
                include={
                    "source",
                    "language",
                    "category",
                    "date_from",
                    "date_to",
                    "recency",
                }
            ),
        )


class ResearchStep(BaseModel):
    tool: Literal["search_news", "search_by_date", "get_article", "compare_sources"]
    arguments: dict
    status: Literal["ok", "empty", "error", "stale"]
    article_ids: list[int] = Field(default_factory=list)
    observation: str


class ResearchSection(BaseModel):
    title: str
    answer: str
    citation_ids: list[int]
    status: Literal["answered", "insufficient_information"]
    reason: str | None = None
    context_hash: str


class ResearchResponse(BaseModel):
    question: str
    status: Literal["answered", "partial", "insufficient_information"]
    answer: str
    sections: list[ResearchSection]
    sources: list[ChatSource]
    plan: list[str]
    planner: Literal["local_model", "fallback", "not_needed"]
    steps: list[ResearchStep]
    limitations: list[str]
    time_intent: TimeIntent
    as_of: datetime
    model_calls: int
    search_calls: int
    tool_calls: int
    elapsed_seconds: float
    stop_reason: Literal[
        "completed",
        "budget_exhausted",
        "dependency_error",
        "no_evidence",
        "evidence_changed",
    ]
    llm_version: str | None = None
    reranking_model_versions: list[str] = Field(default_factory=list)
    translation_model_versions: list[str] = Field(default_factory=list)
    policy_version: str = "research-v1"


class FollowUpPlan(BaseModel):
    """The model can propose queries, never functions, filters, IDs or executable code."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    queries: list[str] = Field(max_length=2)

    @field_validator("queries")
    @classmethod
    def bounded_queries(cls, values):
        for value in values:
            if not 1 <= len(value) <= 300 or not any(c.isalnum() for c in value):
                raise ValueError("Invalid research query")
        return values
