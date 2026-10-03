from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StrictStr, model_validator

from backend.app.schemas.search import SearchRequest
from backend.app.schemas.temporal import TimeIntent
from nlp.multilingual.language import retrieval_question


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    answer_language: Literal["en", "ms", "zh"] | None = None
    recency: Literal["auto", "off", "latest"] = "auto"
    source: str | None = None
    language: str | None = None
    category: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    conversation_id: UUID | None = None
    expected_revision: int | None = Field(default=None, ge=0)
    follow_up: bool | None = None
    reset_context: bool = False

    @model_validator(mode="after")
    def valid_search(self):
        if (self.conversation_id is None) != (self.expected_revision is None):
            raise ValueError("conversation_id and expected_revision must be supplied together")
        if self.reset_context and self.follow_up:
            raise ValueError("reset_context conflicts with follow_up")
        if self.conversation_id is None and (self.follow_up is not None or self.reset_context):
            raise ValueError("Conversation controls require a conversation_id")
        self.search_request()
        return self

    def search_request(self) -> SearchRequest:
        values = self.model_dump(
            include={"recency", "source", "language", "category", "date_from", "date_to"}
        )
        return SearchRequest(
            q=retrieval_question(self.question), method="hybrid", rerank=True, limit=5, **values
        )


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_ids: list[StrictStr] = Field(max_length=5)


class ChatSource(BaseModel):
    citation_id: int
    article_id: int
    title: str
    source: str
    published_at: datetime | None
    retrieved_at: datetime | None = None
    url: HttpUrl
    excerpts: list[str]
    translations: list[str | None] = Field(default_factory=list)


class ChatResponse(BaseModel):
    question: str
    query_language: Literal["en", "ms", "zh"] = "en"
    answer_language: Literal["en", "ms", "zh"] = "en"
    language_detection_fallback: bool = False
    answer: str
    status: Literal["answered", "insufficient_information", "needs_clarification"]
    sources: list[ChatSource] = Field(default_factory=list)
    reason: (
        Literal[
            "no_evidence",
            "model_abstained",
            "invalid_model_output",
            "unsupported_fact",
            "missing_conversation_context",
        ]
        | None
    ) = None
    context_hash: str
    llm_version: str | None = None
    translation_model_version: str | None = None
    reranking_model_version: str | None = None
    time_intent: TimeIntent | None = None
    conversation_id: UUID | None = None
    conversation_revision: int | None = None
    resolved_question: str | None = None
    follow_up: bool = False
    history_turns_used: int = 0
    memory_used: bool = False
