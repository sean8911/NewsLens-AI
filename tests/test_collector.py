import logging

import pytest
import requests

from ingestion.collector import RSSCollector
from ingestion.http import FeedFetchError


@pytest.mark.parametrize(
    "failure",
    [
        requests.Timeout(),
        requests.ConnectionError(),
        requests.HTTPError(),
        FeedFetchError(),
        b"<html>Error</html>",
    ],
)
def test_failure_does_not_stop_following_feed(failure, feed, rss_bytes, caplog):
    bad = feed.model_copy(update={"id": "bad", "url": "https://example.com/bad"})

    class FakeFetcher:
        def fetch(self, url):
            if url.endswith("/bad"):
                if isinstance(failure, Exception):
                    raise failure
                return failure
            return rss_bytes

    with caplog.at_level(logging.INFO):
        result = RSSCollector(FakeFetcher()).collect([bad, feed])
    assert result.successful_feeds == ["example"]
    assert list(result.failed_feeds) == ["bad"]
    assert len(result.articles) == 3
    assert result.entries_found == 5
    assert result.skipped_entries == 2
    assert "feed_failed feed=bad" in caplog.text
    assert "feed_success feed=example" in caplog.text
    assert "collection_complete" in caplog.text


def test_disabled_feed_not_requested(feed):
    class NoFetcher:
        def fetch(self, url):
            pytest.fail("disabled feed was fetched")

    result = RSSCollector(NoFetcher()).collect([feed.model_copy(update={"enabled": False})])
    assert result.successful_feeds == []
