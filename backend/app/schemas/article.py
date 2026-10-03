from datetime import UTC, datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, field_validator


class Article(BaseModel):
    """An RSS record, not a database model. Descriptions may contain feed HTML."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1)
    description: str | None = None
    url: HttpUrl
    source: str = Field(min_length=1)
    author: str | None = None
    published_at: AwareDatetime | None = None
    retrieved_at: AwareDatetime
    category: str
    rss_categories: list[str] = Field(default_factory=list)

    @field_validator("published_at", "retrieved_at")
    @classmethod
    def normalize_timezone(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(UTC) if value is not None else None
