from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import MagicMock

import pytest
import requests

from backend.app.core.config import Settings
from ingestion.http import FeedFetchError, RSSFetcher


def test_retries_transient_http_failure_with_real_adapter():
    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def do_GET(self):
            Handler.calls += 1
            self.send_response(503 if Handler.calls == 1 else 200)
            self.end_headers()
            self.wfile.write(b"feed")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with RSSFetcher(Settings(_env_file=None, retry_count=1)) as fetcher:
            fetcher.session.trust_env = False
            assert fetcher.fetch(f"http://127.0.0.1:{server.server_port}/rss") == b"feed"
        assert Handler.calls == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_response_limit_and_timeout_settings():
    with RSSFetcher(Settings(_env_file=None, max_feed_bytes=5)) as fetcher:
        session = MagicMock()
        fetcher.session.close()
        fetcher.session = session
        response = session.get.return_value.__enter__.return_value
        response.iter_content.return_value = [b"123", b"456"]
        with pytest.raises(FeedFetchError):
            fetcher.fetch("https://example.com/rss")
        assert session.get.call_args.kwargs["timeout"] == (5, 20)
        session.get.return_value.__exit__.assert_called_once()
    session.close.assert_called_once()


def test_http_errors_propagate_for_collector_to_handle():
    with RSSFetcher(Settings(_env_file=None)) as fetcher:
        original = fetcher.session
        fetcher.session = MagicMock()
        original.close()
        response = fetcher.session.get.return_value.__enter__.return_value
        response.raise_for_status.side_effect = requests.HTTPError()
        with pytest.raises(requests.HTTPError):
            fetcher.fetch("https://example.com/rss")
        response.iter_content.assert_not_called()
