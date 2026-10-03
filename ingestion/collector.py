import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from requests import RequestException

from backend.app.schemas.article import Article
from ingestion.cleaners.processor import ProcessingResult
from ingestion.feeds.config import FeedConfig
from ingestion.http import FeedFetchError
from ingestion.parsers.rss import FeedParseError, parse_feed

logger = logging.getLogger(__name__)


class Fetcher(Protocol):
    def fetch(self, url: str) -> bytes: ...


@dataclass
class CollectionResult:
    articles: list[Article] = field(default_factory=list)
    successful_feeds: list[str] = field(default_factory=list)
    failed_feeds: dict[str, str] = field(default_factory=dict)
    entries_found: int = 0
    skipped_entries: int = 0
    records: list[ProcessingResult] = field(default_factory=list)
    changed_entries: int = 0


class RSSCollector:
    def __init__(self, fetcher: Fetcher, *, content_min_chars: int = 300) -> None:
        self.fetcher = fetcher
        self.content_min_chars = content_min_chars

    def collect(self, feeds: Iterable[FeedConfig]) -> CollectionResult:
        result = CollectionResult()
        started = perf_counter()
        for feed in feeds:
            if not feed.enabled:
                continue
            try:
                content = self.fetcher.fetch(str(feed.url))
                parsed = parse_feed(
                    content, feed, datetime.now(UTC), content_min_chars=self.content_min_chars
                )
            except (RequestException, FeedFetchError, FeedParseError) as exc:
                # Keep failure metadata useful without exposing request URLs/credentials.
                error = type(exc).__name__
                result.failed_feeds[feed.id] = error
                logger.error("feed_failed feed=%s error=%s", feed.id, error)
                continue
            result.successful_feeds.append(feed.id)
            cleaned = [record.article for record in parsed.records if record.article is not None]
            rejected = len(parsed.records) - len(cleaned)
            result.articles.extend(cleaned)
            result.records.extend(parsed.records)
            result.changed_entries += sum(
                record.article is not None and bool(record.issues) for record in parsed.records
            )
            result.entries_found += parsed.entries_found
            result.skipped_entries += rejected
            for index, record in enumerate(parsed.records):
                if record.issues:
                    logger.log(
                        logging.WARNING if record.article is None else logging.DEBUG,
                        "article_processing feed=%s entry_index=%d accepted=%s issues=%s",
                        feed.id,
                        index,
                        record.article is not None,
                        ",".join(record.issues),
                    )
            logger.info(
                "feed_success feed=%s entries=%d articles=%d skipped=%d malformed=%s",
                feed.id,
                parsed.entries_found,
                len(cleaned),
                rejected,
                parsed.malformed,
            )
        logger.info(
            "collection_complete successful_feeds=%d failed_feeds=%d entries=%d "
            "articles=%d skipped=%d changed=%d duration_seconds=%.2f",
            len(result.successful_feeds),
            len(result.failed_feeds),
            result.entries_found,
            len(result.articles),
            result.skipped_entries,
            result.changed_entries,
            perf_counter() - started,
        )
        return result
