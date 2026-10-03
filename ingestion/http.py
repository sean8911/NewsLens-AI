import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.app.core.config import Settings


class FeedFetchError(ValueError):
    """A feed response exceeds the configured size limit."""


class RSSFetcher:
    """Owns and closes a reusable requests session; no parsing or persistence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": settings.user_agent,
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
            }
        )
        retries = Retry(
            total=settings.retry_count,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            retry_after_max=30,
        )
        for scheme in ("https://", "http://"):
            self.session.mount(scheme, HTTPAdapter(max_retries=retries))

    def fetch(self, url: str) -> bytes:
        with self.session.get(
            url,
            timeout=(
                self.settings.connect_timeout_seconds,
                self.settings.read_timeout_seconds,
            ),
            stream=True,
        ) as response:
            response.raise_for_status()
            chunks = []
            size = 0
            for chunk in response.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > self.settings.max_feed_bytes:
                    raise FeedFetchError("Feed exceeds NEWSLENS_MAX_FEED_BYTES")
                chunks.append(chunk)
            return b"".join(chunks)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "RSSFetcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
