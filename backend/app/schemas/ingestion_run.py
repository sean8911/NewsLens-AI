from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RunReport(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    started_at: datetime
    finished_at: datetime | None
    status: Literal["running", "succeeded", "partial", "failed", "interrupted"]
    feeds_checked: int
    successful_feeds: list[str]
    failed_feeds: dict[str, str]
    articles_found: int
    articles_accepted: int
    articles_changed: int
    inserted: int
    duplicates: int
    rejected: int
    duration_seconds: float | None
    error_type: str | None


class UpdateStatus(BaseModel):
    last_successful_news_update: datetime | None
    last_complete_news_update: datetime | None
    recent_runs: list[RunReport]
