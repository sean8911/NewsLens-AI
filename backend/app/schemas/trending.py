from datetime import UTC, timedelta
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, model_validator


class TrendingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    as_of: AwareDatetime | None = None
    window_hours: int = Field(default=24, ge=1, le=168)
    group_by: Literal["topic", "entity_topic"] = "topic"
    min_sources: int = Field(default=2, ge=2, le=20)
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def valid_window(self):
        if self.as_of is not None:
            try:
                self.as_of.astimezone(UTC) - timedelta(hours=2 * self.window_hours)
            except (OverflowError, ValueError) as exc:
                raise ValueError("Timestamp cannot represent both trend windows") from exc
        return self


class TrendEvidence(BaseModel):
    article_id: int
    title: str
    source: str
    url: HttpUrl
    published_at: AwareDatetime


class TrendEntityStat(BaseModel):
    name: str
    entity_type: str
    article_count: int
    source_count: int


class TrendingItem(BaseModel):
    id: str
    label: str
    topic: str
    entity: TrendEntityStat | None = None
    score: float
    components: dict[str, float]
    article_count: int
    previous_article_count: int
    source_count: int
    previous_source_count: int
    sources: list[str]
    articles_per_hour: float
    previous_articles_per_hour: float
    growth_percent: float | None
    state: Literal["new", "growing", "steady", "declining", "baseline_unavailable"]
    capped_activity: int
    previous_capped_activity: int
    topic_share: float
    previous_topic_share: float | None
    entities: list[TrendEntityStat]
    evidence: list[TrendEvidence]
    previous_evidence: list[TrendEvidence]


class TrendingResponse(BaseModel):
    status: Literal["ready", "no_trends"]
    as_of: AwareDatetime
    current_start: AwareDatetime
    previous_start: AwareDatetime
    window_hours: int
    group_by: Literal["topic", "entity_topic"]
    count: int
    total_candidates: int
    current_articles: int
    previous_articles: int
    articles_scanned: int
    excluded: dict[str, int]
    articles_without_current_entities: int
    policy_version: str
    input_hash: str
    weights: dict[str, float]
    results: list[TrendingItem]
    scope: list[str]
