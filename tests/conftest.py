import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingestion.feeds.config import FeedConfig


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """A developer's .env or custom feed settings must not affect offline tests."""
    for name in os.environ:
        if name.upper().startswith("NEWSLENS_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def feed():
    return FeedConfig(
        id="example", source="Example News", url="https://example.com/rss", category="Technology"
    )


@pytest.fixture
def retrieved_at():
    return datetime(2026, 9, 9, 8, 0, tzinfo=UTC)


@pytest.fixture
def rss_bytes():
    return (Path(__file__).parent / "fixtures" / "news.xml").read_bytes()
