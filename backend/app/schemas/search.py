from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.schemas.temporal import TemporalDetails


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    q: str = Field(min_length=1, max_length=2000, description="Natural-language or keyword query")
    method: Literal["semantic", "lexical", "hybrid"] = "semantic"
    recency: Literal["auto", "off", "latest"] = "auto"
    limit: int = Field(default=10, ge=1, le=50)
    rerank: bool = Field(
        default=False, description="Rerank candidates with the local cross-encoder"
    )
    source: str | None = Field(default=None, min_length=1, max_length=200)
    language: str | None = Field(default=None, pattern=r"^[a-z]{2,16}$")
    category: str | None = Field(default=None, min_length=1, max_length=200)
    date_from: date | None = Field(default=None, description="Inclusive UTC publication date")
    date_to: date | None = Field(default=None, description="Inclusive UTC publication date")

    @field_validator("q")
    @classmethod
    def meaningful_query(cls, value: str) -> str:
        if not any(char.isalnum() for char in value):
            raise ValueError("Query must contain a letter or number")
        return value

    @model_validator(mode="after")
    def ordered_dates(self) -> Self:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self


class SearchHit(BaseModel):
    article_id: int
    title: str
    source: str
    published_at: datetime | None
    relevance_score: float = Field(allow_inf_nan=False)
    summary: str | None
    url: str
    language: str | None
    category: str


class FusionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_limit: int = Field(default=50, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    semantic_weight: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)
    lexical_weight: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)


class FusionContribution(BaseModel):
    rank: int
    score: float
    contribution: float


class FusionExplanation(BaseModel):
    article_id: int
    semantic: FusionContribution | None = None
    lexical: FusionContribution | None = None


class FusionDetails(BaseModel):
    algorithm: Literal["rrf"] = "rrf"
    config: FusionConfig
    semantic_candidates: int
    lexical_candidates: int
    explanations: list[FusionExplanation]


class RerankingScore(BaseModel):
    article_id: int
    retrieval_rank: int
    retrieval_score: float = Field(allow_inf_nan=False)
    reranking_rank: int
    reranking_score: float = Field(allow_inf_nan=False)
    input_hash: str


class RerankingDetails(BaseModel):
    algorithm: Literal["cross-encoder"] = "cross-encoder"
    model_version: str
    input_policy: str
    candidate_limit: int
    candidate_count: int
    scores: list[RerankingScore]


class SearchResponse(BaseModel):
    query: str
    query_language: Literal["en", "ms", "zh"] | None = None
    language_detection_fallback: bool = False
    method: Literal["semantic", "lexical", "hybrid"] = "semantic"
    collection: str | None = None
    model_version: str | None = None
    count: int
    results: list[SearchHit]
    fusion: FusionDetails | None = None
    reranking: RerankingDetails | None = None
    temporal: TemporalDetails | None = None
