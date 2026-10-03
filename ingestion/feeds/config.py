from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter


class FeedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    source: str = Field(min_length=1)
    url: HttpUrl
    category: str = Field(min_length=1)
    enabled: bool = True


def load_feeds(path: Path | None = None) -> list[FeedConfig]:
    raw = (
        path.read_text(encoding="utf-8")
        if path is not None
        else files("ingestion.feeds").joinpath("defaults.json").read_text(encoding="utf-8")
    )
    feeds = TypeAdapter(list[FeedConfig]).validate_json(raw)
    ids = [feed.id for feed in feeds]
    if len(ids) != len(set(ids)):
        raise ValueError("Feed IDs must be unique")
    enabled = [feed for feed in feeds if feed.enabled]
    if not enabled:
        raise ValueError("Configure at least one enabled feed")
    return enabled
