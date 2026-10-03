import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import feedparser
from pydantic import ValidationError

from backend.app.schemas.article import Article
from ingestion.cleaners.processor import ArticleProcessor, ProcessingResult, snapshot
from ingestion.extraction.service import extract_rss
from ingestion.feeds.config import FeedConfig

logger = logging.getLogger(__name__)


class FeedParseError(ValueError):
    """The response does not contain a usable RSS/Atom feed."""


@dataclass
class ParsedFeed:
    articles: list[Article]
    entries_found: int
    skipped: int
    malformed: bool
    records: list[ProcessingResult]


def optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def publication_date(entry: dict[str, Any]) -> datetime | None:
    # Do not mislabel Atom's updated timestamp as the original publication date.
    parsed = entry.get("published_parsed") if "published_parsed" in entry else None
    if parsed is not None:
        try:
            return datetime(*parsed[:6], tzinfo=UTC)
        except (ValueError, TypeError, OverflowError):
            pass
    if "published" in entry:
        logger.warning("invalid_publication_date; storing null")
    return None


def normalize_entry(entry: dict[str, Any], feed: FeedConfig, retrieved_at: datetime) -> Article:
    categories = [term for tag in entry.get("tags", []) if (term := optional_text(tag.get("term")))]
    return Article(
        title=optional_text(entry.get("title")) or "",
        description=optional_text(entry.get("summary")) or optional_text(entry.get("description")),
        url=entry.get("link", ""),
        source=feed.source,
        author=optional_text(entry.get("author")),
        published_at=publication_date(entry),
        retrieved_at=retrieved_at,
        category=feed.category,
        rss_categories=list(dict.fromkeys(categories)),
    )


def parse_feed(
    content: bytes, feed: FeedConfig, retrieved_at: datetime, *, content_min_chars: int = 300
) -> ParsedFeed:
    # Passing bytes keeps HTTP concerns in the fetcher and respects XML encoding declarations.
    parsed = feedparser.parse(content, resolve_relative_uris=False, sanitize_html=False)
    malformed = bool(parsed.get("bozo"))
    if not parsed.get("version") or (malformed and not parsed.entries):
        raise FeedParseError("Response is not a usable RSS/Atom feed")
    if malformed:
        logger.warning("malformed_feed feed=%s; attempting recovered entries", feed.id)
    articles = []
    skipped = 0
    records = []
    for index, entry in enumerate(parsed.entries):
        raw_json = snapshot(
            {
                "format": "feedparser_entry",
                "feed": feed.model_dump(mode="json"),
                "retrieved_at": retrieved_at.isoformat(),
                "entry": dict(entry),
            }
        )
        try:
            article = normalize_entry(entry, feed, retrieved_at)
            articles.append(article)
            record = ArticleProcessor().process(article, raw_json=raw_json)
            extract_rss(record, content_min_chars)
            records.append(record)
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            skipped += 1
            # Do not dump entire third-party records into logs.
            logger.warning("invalid_entry feed=%s entry_index=%d; skipped", feed.id, index)
            issues = (
                [
                    "invalid_" + str(error["loc"][0])
                    for error in exc.errors(include_input=False, include_url=False)
                ]
                if isinstance(exc, ValidationError)
                else ["normalization_failed"]
            )
            records.append(ProcessingResult(raw_json, None, issues))
    return ParsedFeed(articles, len(parsed.entries), skipped, malformed, records)
