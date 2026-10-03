from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from ingestion.parsers.rss import FeedParseError, normalize_entry, parse_feed, publication_date


def test_rss_schema_optional_fields_and_bad_entries(feed, retrieved_at, rss_bytes, caplog):
    result = parse_feed(rss_bytes, feed, retrieved_at)
    assert result.entries_found == 5
    assert result.skipped == 2
    assert len(result.articles) == 3
    article, minimal, invalid_date = result.articles
    assert article.title == "Example AI headline"
    assert article.source == "Example News"
    assert str(article.url) == "https://example.com/news/ai?ref=rss"
    assert article.description == "<p>Research &amp; technology update.</p>"
    assert "Example Reporter" in article.author
    assert article.published_at == datetime(2026, 9, 9, 6, 30, tzinfo=UTC)
    assert article.retrieved_at == retrieved_at
    assert article.category == "Technology"
    assert article.rss_categories == ["AI", "Technology"]
    assert minimal.description is None
    assert minimal.author is None
    assert minimal.published_at is None
    assert minimal.rss_categories == []
    assert invalid_date.published_at is None
    assert "invalid_publication_date" in caplog.text
    assert "invalid_entry" in caplog.text
    assert article.model_dump(mode="json")["retrieved_at"] == "2026-09-09T08:00:00Z"


def test_atom_preserves_unicode_and_selects_article_link(feed, retrieved_at):
    content = (Path(__file__).parent / "fixtures" / "atom.xml").read_bytes()
    first, updated_only = parse_feed(content, feed, retrieved_at).articles
    assert first.title == "Berita AI di Malaysia — 马来西亚"
    assert str(first.url) == "https://example.com/story"
    assert first.author == "Example Writer"
    assert first.published_at == datetime(2026, 9, 9, 6, tzinfo=UTC)
    assert first.rss_categories == ["Malaysia"]
    assert updated_only.published_at is None


@pytest.mark.parametrize("content", [b"", b"<html>Access denied</html>", b"not XML", b"<rss>"])
def test_unusable_feed_fails(content, feed, retrieved_at):
    with pytest.raises(FeedParseError):
        parse_feed(content, feed, retrieved_at)


def test_recoverable_malformed_feed_keeps_valid_entries(feed, retrieved_at, caplog):
    content = (
        b'<rss version="2.0"><channel><title>Feed</title><item><title>Recovered</title>'
        b"<link>https://example.com/recovered</link></item>"
    )
    result = parse_feed(content, feed, retrieved_at)
    assert result.malformed
    assert result.articles[0].title == "Recovered"
    assert "malformed_feed" in caplog.text


def test_valid_empty_feed_is_success(feed, retrieved_at):
    result = parse_feed(
        b'<rss version="2.0"><channel><title>Empty</title></channel></rss>', feed, retrieved_at
    )
    assert result.articles == []
    assert not result.malformed


@pytest.mark.parametrize("link", ["", "relative/path", "ftp://example.com/file", "bad url"])
def test_invalid_urls_rejected(link, feed, retrieved_at):
    with pytest.raises(ValidationError):
        normalize_entry({"title": "Title", "link": link}, feed, retrieved_at)


@pytest.mark.parametrize("title", [None, "", "  "])
def test_missing_title_rejected(title, feed, retrieved_at):
    with pytest.raises(ValidationError):
        normalize_entry({"title": title, "link": "https://example.com/a"}, feed, retrieved_at)


def test_impossible_parsed_date_is_null():
    assert publication_date({"published_parsed": (2026, 13, 40, 0, 0, 0)}) is None


def test_schema_rejects_naive_retrieval_time(feed):
    with pytest.raises(ValidationError):
        normalize_entry(
            {"title": "News", "link": "https://example.com/a"}, feed, datetime(2026, 9, 9)
        )
