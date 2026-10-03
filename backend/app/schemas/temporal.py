from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TimeIntent(BaseModel):
    kind: Literal["none", "latest", "historical", "explicit"]
    as_of: datetime
    retrieval_query: str
    date_from: date | None = None
    date_to: date | None = None
    expression: str | None = None
    reason: str


class RecencyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    decay: Literal["exponential", "gaussian", "hyperbolic"] = "exponential"
    half_life_days: float = Field(default=7, gt=0, le=3650, allow_inf_nan=False)
    strength: float = Field(default=0.65, ge=0, le=1, allow_inf_nan=False)
    reranking_weight: float = Field(default=0.85, ge=0, le=1, allow_inf_nan=False)


class TemporalScore(BaseModel):
    article_id: int
    prior_rank: int
    prior_score: float = Field(allow_inf_nan=False)
    relevance_component: float
    age_days: float | None
    date_status: Literal["known", "missing", "future"]
    decay_factor: float
    final_score: float = Field(allow_inf_nan=False)


class TemporalDetails(BaseModel):
    intent: TimeIntent
    applied: bool
    config: RecencyConfig
    scores: list[TemporalScore] = Field(default_factory=list)
