import json

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from ingestion.cli import main
from ingestion.feeds.config import load_feeds
from ingestion.http import RSSFetcher


def test_bundled_feeds_cover_required_topics():
    feeds = load_feeds()
    assert 5 <= len(feeds) <= 10
    assert {
        "Artificial Intelligence",
        "Technology",
        "Business",
        "Science",
        "Malaysia",
        "World News",
    } <= {feed.category for feed in feeds}


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"id": "bad"}],
        [
            {"id": "same", "source": "News", "url": "https://example.com", "category": "World"},
            {"id": "same", "source": "News", "url": "https://example.com", "category": "World"},
        ],
    ],
)
def test_bad_feed_configuration_rejected(tmp_path, records):
    path = tmp_path / "feeds.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError):
        load_feeds(path)


def test_environment_overrides_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("NEWSLENS_RETRY_COUNT=1\n", encoding="utf-8")
    monkeypatch.setenv("NEWSLENS_RETRY_COUNT", "3")
    assert Settings(_env_file=env).retry_count == 3


def test_invalid_timeout_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, connect_timeout_seconds=0)


def test_cli_outputs_only_article_json_on_stdout(monkeypatch, capsys, rss_bytes):
    monkeypatch.setattr(RSSFetcher, "fetch", lambda self, url: rss_bytes)
    assert main(["--feed", "bbc-world"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 3
    assert records[0]["source"] == "BBC News"


def test_cli_all_failed_is_nonzero(monkeypatch):
    monkeypatch.setattr(RSSFetcher, "fetch", lambda self, url: b"not a feed")
    assert main(["--feed", "bbc-world"]) == 1


def test_cli_unknown_feed_is_configuration_error():
    with pytest.raises(SystemExit) as exc:
        main(["--feed", "no-such-feed"])
    assert exc.value.code == 2


def test_cli_lists_feeds_without_network(monkeypatch, capsys):
    def fail(*args):
        pytest.fail("list should not use network")

    monkeypatch.setattr(RSSFetcher, "fetch", fail)
    assert main(["--list-feeds"]) == 0
    assert "guardian-malaysia" in capsys.readouterr().out
