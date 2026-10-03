"""Versioned, offline evaluation data contracts; no model execution or database writes."""

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from backend.app.schemas.search import SearchRequest

QUESTION_TYPES = (
    "factual",
    "latest_update",
    "entity",
    "timeline",
    "multi_source_comparison",
    "difficult_retrieval",
    "no_answer",
)
DEFAULT_DATASET = Path(__file__).parent / "fixtures" / "news-evaluation-v1.json"
ArticleId = Annotated[int, Field(strict=True, gt=0)]


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SnapshotArticle(StrictModel):
    id: ArticleId
    title: str = Field(min_length=1, max_length=10000)
    description: str | None = Field(default=None, max_length=100000)
    content: str | None = Field(default=None, max_length=1000000)
    source: str = Field(min_length=1, max_length=200)
    url: HttpUrl
    published_at: datetime | None
    category: str
    language: str | None

    @field_validator("published_at")
    @classmethod
    def aware_date(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError("Publication timestamps must be timezone-aware")
        return value.astimezone(UTC) if value else None


def corpus_hash(articles: list[SnapshotArticle]) -> str:
    return fingerprint(
        [article.model_dump(mode="json") for article in sorted(articles, key=lambda a: a.id)]
    )


class TimeContext(StrictModel):
    as_of: datetime
    date_from: date | None = None
    date_to: date | None = None
    interpretation: Literal["snapshot", "latest", "publication_timeline", "date_window"]
    note: str = Field(min_length=10, max_length=1000)

    @field_validator("as_of")
    @classmethod
    def aware_as_of(cls, value):
        if value.tzinfo is None:
            raise ValueError("Reference time must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("Date window is reversed")
        return self


class DatasetFilters(StrictModel):
    source: str | None = Field(default=None, min_length=1, max_length=200)
    category: str | None = Field(default=None, min_length=1, max_length=200)
    language: str | None = Field(default=None, pattern=r"^[a-z]{2,16}$")


class AnswerEvidence(StrictModel):
    article_id: ArticleId
    field: Literal["title", "description", "content"]
    start: int = Field(strict=True, ge=0)
    end: int = Field(strict=True, gt=0)
    quote: str = Field(min_length=1, max_length=5000)


class LatestSelection(StrictModel):
    """Transparent annotation rule, not a retrieval algorithm or model input."""

    title_contains_all: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("title_contains_all")
    @classmethod
    def nonblank_terms(cls, values):
        if any(not value.strip() or len(value) > 100 for value in values):
            raise ValueError("Latest-selection terms must be nonblank and bounded")
        return values


class NewsQuestion(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,79}$")
    question: str = Field(min_length=10, max_length=2000)
    expected_article_ids: list[ArticleId] = Field(max_length=10)
    expected_answer: str = Field(min_length=10, max_length=10000)
    question_type: Literal[
        "factual",
        "latest_update",
        "entity",
        "timeline",
        "multi_source_comparison",
        "difficult_retrieval",
        "no_answer",
    ]
    time_context: TimeContext
    filters: DatasetFilters = Field(default_factory=DatasetFilters)
    evidence: list[AnswerEvidence] = Field(max_length=30)
    related_article_ids: list[ArticleId] = Field(default_factory=list, max_length=10)
    topic_group: str = Field(pattern=r"^[a-z][a-z0-9-]{2,79}$")
    annotation_rationale: str = Field(min_length=20, max_length=2000)
    latest_selection: LatestSelection | None = None

    @model_validator(mode="after")
    def valid_case(self) -> Self:
        self.search_request()
        if len(set(self.expected_article_ids)) != len(self.expected_article_ids):
            raise ValueError("Expected article IDs must be unique")
        if len(set(self.related_article_ids)) != len(self.related_article_ids):
            raise ValueError("Related article IDs must be unique")
        if set(self.expected_article_ids) & set(self.related_article_ids):
            raise ValueError("Expected and related IDs must be distinct")
        if self.question_type == "no_answer":
            if self.expected_article_ids or self.evidence or self.latest_selection:
                raise ValueError("No-answer cases cannot contain positive answer evidence")
        elif not self.expected_article_ids or not self.evidence:
            raise ValueError("Answerable cases need article IDs and supporting excerpts")
        if (self.question_type == "latest_update") != (self.latest_selection is not None):
            raise ValueError("Only latest-update cases require a latest selection rule")
        if self.question_type == "latest_update" and self.time_context.interpretation != "latest":
            raise ValueError("Latest updates require latest time context")
        if self.question_type == "timeline":
            if (
                len(self.expected_article_ids) < 2
                or self.time_context.interpretation != "publication_timeline"
            ):
                raise ValueError("Timelines require multiple articles and publication-time context")
        return self

    def search_request(self) -> SearchRequest:
        return SearchRequest(
            q=self.question,
            **self.filters.model_dump(),
            date_from=self.time_context.date_from,
            date_to=self.time_context.date_to,
            recency="off",  # Evaluation runner must use the explicit frozen reference time.
        )

    def question_input(self) -> dict:
        # Gold labels, latest-selection rules and related IDs must never enter retrieval/prompts.
        return self.model_dump(mode="json", include={"id", "question", "filters", "time_context"})


def in_scope(article: SnapshotArticle, case: NewsQuestion) -> bool:
    filters, context = case.filters, case.time_context
    if filters.source and article.source != filters.source:
        return False
    if filters.category and article.category != filters.category:
        return False
    if filters.language and (article.language or "unknown") != filters.language:
        return False
    published = article.published_at
    if published and published > context.as_of:
        return False
    if context.date_from and (not published or published.date() < context.date_from):
        return False
    if context.date_to and (not published or published.date() > context.date_to):
        return False
    return True


class NewsEvaluationDataset(StrictModel):
    schema_version: Literal["1.0"]
    name: str = Field(min_length=1)
    dataset_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    purpose: Literal["development"]
    annotation_method: str = Field(min_length=20)
    scope_note: str = Field(min_length=20)
    captured_at: datetime
    corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    articles: list[SnapshotArticle] = Field(min_length=1, max_length=5000)
    questions: list[NewsQuestion] = Field(min_length=50, max_length=100)

    @field_validator("captured_at")
    @classmethod
    def aware_capture(cls, value):
        if value.tzinfo is None:
            raise ValueError("Capture time must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        articles = {article.id: article for article in self.articles}
        if len(articles) != len(self.articles):
            raise ValueError("Snapshot article IDs must be unique")
        if len({str(article.url) for article in self.articles}) != len(self.articles):
            raise ValueError("Snapshot article URLs must be unique")
        if corpus_hash(self.articles) != self.corpus_sha256:
            raise ValueError("Corpus fingerprint mismatch")
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("Question IDs must be unique")
        normalized = {re.sub(r"\W+", "", q.question.casefold()) for q in self.questions}
        if len(normalized) != len(self.questions):
            raise ValueError("Duplicate normalized questions")
        counts = Counter(q.question_type for q in self.questions)
        if any(counts[kind] < 5 for kind in QUESTION_TYPES):
            raise ValueError("Each of the seven question types requires at least five cases")
        for case in self.questions:
            expected = set(case.expected_article_ids)
            if (expected | set(case.related_article_ids)) - articles.keys():
                raise ValueError(f"{case.id}: unknown article reference")
            if any(not in_scope(articles[id_], case) for id_ in expected):
                raise ValueError(f"{case.id}: gold article is outside the question scope")
            if {item.article_id for item in case.evidence} != expected:
                raise ValueError(f"{case.id}: evidence must cover exactly the expected articles")
            seen = set()
            for item in case.evidence:
                key = (item.article_id, item.field, item.start, item.end)
                if key in seen:
                    raise ValueError(f"{case.id}: duplicate evidence span")
                seen.add(key)
                original = getattr(articles[item.article_id], item.field) or ""
                if (
                    not 0 <= item.start < item.end <= len(original)
                    or original[item.start : item.end] != item.quote
                ):
                    raise ValueError(f"{case.id}: evidence does not match the original field slice")
            if case.question_type == "multi_source_comparison":
                publishers = {articles[id_].source.strip().casefold() for id_ in expected}
                if len(publishers) < 2:
                    raise ValueError(f"{case.id}: comparison requires distinct publishers")
            if case.question_type == "timeline":
                rows = [articles[id_] for id_ in case.expected_article_ids]
                if any(row.published_at is None for row in rows):
                    raise ValueError(f"{case.id}: timeline requires publication timestamps")
                if rows != sorted(rows, key=lambda row: (row.published_at, row.id)):
                    raise ValueError(f"{case.id}: timeline IDs must be in publication order")
            if case.latest_selection is not None:
                candidates = [
                    article
                    for article in self.articles
                    if article.published_at is not None
                    and in_scope(article, case)
                    and all(
                        term.casefold() in article.title.casefold()
                        for term in case.latest_selection.title_contains_all
                    )
                ]
                if not candidates:
                    raise ValueError(f"{case.id}: latest selection has no eligible candidates")
                latest = max(article.published_at for article in candidates)
                if expected != {
                    article.id for article in candidates if article.published_at == latest
                }:
                    raise ValueError(f"{case.id}: latest labels are stale or incomplete")
        return self

    def summary(self) -> dict:
        return {
            "name": self.name,
            "dataset_version": self.dataset_version,
            "purpose": self.purpose,
            "question_count": len(self.questions),
            "question_types": dict(
                sorted(Counter(q.question_type for q in self.questions).items())
            ),
            "article_count": len(self.articles),
            "corpus_sha256": self.corpus_sha256,
            "dataset_sha256": fingerprint(self.model_dump(mode="json")),
            "publishers": dict(sorted(Counter(a.source for a in self.articles).items())),
            "cited_articles": len({id_ for q in self.questions for id_ in q.expected_article_ids}),
            "topic_groups": len({q.topic_group for q in self.questions}),
            "evidence_spans": sum(len(q.evidence) for q in self.questions),
            "no_answer_count": sum(q.question_type == "no_answer" for q in self.questions),
            "model_metrics_measured": False,
        }


def load_dataset(path: Path = DEFAULT_DATASET) -> NewsEvaluationDataset:
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Dataset exceeds 20 MiB")
    return NewsEvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))
