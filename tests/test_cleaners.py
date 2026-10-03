import json
from datetime import UTC, datetime, timedelta

import pytest

from backend.app.schemas.article import Article
from ingestion.cleaners.dates import clean_publication_date
from ingestion.cleaners.processor import ArticleProcessor, snapshot
from ingestion.cleaners.sources import normalize_source
from ingestion.cleaners.text import clean_text
from ingestion.cleaners.urls import clean_url
from ingestion.collector import RSSCollector
from ingestion.parsers.rss import parse_feed


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (" \n\t", None),
        ("<p>AI <b>news</b>.</p><p>Science &amp; business.</p>", "AI news.\n\nScience & business."),
        ("<script>alert(1)</script><style>body{}</style><p>Safe headline</p>", "Safe headline"),
        ("&lt;p&gt;News &amp;amp; science&lt;/p&gt;", "News & science"),
        ("Malaysia\u00a0\u00a0AI\r\n\n\nScience", "Malaysia AI\n\nScience"),
        ("FranÃ§ais café", "Français café"),
        ("马来西亚 AI\x00\u200b", "马来西亚 AI"),
        (
            "<p>A long repeated paragraph about news.</p>" * 2,
            "A long repeated paragraph about news.",
        ),
        ("No.\nNo.", "No.\n\nNo."),
        ("The very very important story", "The very very important story"),
        ("Value < 3 and price > 2", "Value < 3 and price > 2"),
    ],
)
def test_text_cleaning_and_idempotence(raw, expected):
    assert clean_text(raw) == expected
    assert clean_text(expected) == expected


def test_text_keeps_nonadjacent_paragraphs_and_unicode_joiners():
    paragraph = "A repeated paragraph that may be a legitimate quote."
    raw = f"{paragraph}\nA different paragraph.\n{paragraph}"
    assert clean_text(raw).count(paragraph) == 2
    assert "\u200d" in clean_text("👩\u200d💻 works")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://example.com/a?utm_source=rss&id=2&fbclid=x#part",
            "https://example.com/a?id=2#part",
        ),
        (
            "https://example.com/a?id=a%20b&ref=rss&id=c+d&utm_medium=rss",
            "https://example.com/a?id=a%20b&ref=rss&id=c+d",
        ),
        (
            "https://example.com/a?utm_source=x&sig=abc",
            "https://example.com/a?utm_source=x&sig=abc",
        ),
        (
            "https://example.com/a?X-Amz-Signature=x&utm_source=rss",
            "https://example.com/a?X-Amz-Signature=x&utm_source=rss",
        ),
        ("https://www.bbc.co.uk/a?at_medium=RSS&at_campaign=rss", "https://www.bbc.co.uk/a"),
        ("https://notbbc.com/a?at_medium=RSS", "https://notbbc.com/a?at_medium=RSS"),
        ("https://example.com/a?UTM_SOURCE=rss&foo=", "https://example.com/a?foo="),
    ],
)
def test_conservative_url_cleaning(url, expected):
    assert clean_url(url) == expected
    assert clean_url(expected) == expected


@pytest.mark.parametrize(
    "url",
    [
        "bad url",
        "javascript:alert(1)",
        "https://user:secret@example.com",
        "https://example.com/\x00a",
    ],
)
def test_invalid_article_url(url):
    with pytest.raises(ValueError):
        clean_url(url)


def test_exact_source_aliases_can_be_disabled():
    assert normalize_source(" BBC News - Technology ") == "BBC News"
    assert normalize_source("BBC", {}) == "BBC"
    assert normalize_source("BBC News - Made Up") == "BBC News - Made Up"


@pytest.mark.parametrize(
    "raw_date", ["not a date", "Wed, 31 Feb 2026 08:00:00 GMT", "2026-13-01T00:00:00Z"]
)
def test_invalid_date_is_unknown(raw_date, retrieved_at):
    value, issue = clean_publication_date(None, retrieved_at, raw_date=raw_date)
    assert value is None and issue == "invalid_publication_date"


def test_date_bounds_and_timezone(retrieved_at):
    assert clean_publication_date(retrieved_at + timedelta(days=2), retrieved_at)[0] is None
    assert clean_publication_date(datetime(1800, 1, 1, tzinfo=UTC), retrieved_at)[0] is None
    assert clean_publication_date(None, retrieved_at, raw_date="2026-09-09T08:00:00")[1] == (
        "ambiguous_publication_timezone"
    )
    assert clean_publication_date(None, retrieved_at, raw_date="2026-09-09T14:00:00+08:00")[0] == (
        datetime(2026, 9, 9, 6, tzinfo=UTC)
    )
    assert clean_publication_date(None, retrieved_at) == (None, None)


def test_processor_keeps_raw_and_drops_duplicated_description(retrieved_at):
    article = Article(
        title="<b>AI news</b>",
        description="AI news",
        url="https://example.com?utm_source=x",
        source="BBC",
        category="Technology",
        retrieved_at=retrieved_at,
    )
    result = ArticleProcessor().process(article)
    assert result.article.title == "AI news"
    assert result.article.description is None
    assert "description_duplicates_title" in result.issues
    assert json.loads(result.raw_json)["article"]["title"] == "<b>AI news</b>"
    assert ArticleProcessor().process(result.article).article == result.article


def test_parser_and_collector_keep_rejected_inputs(feed, retrieved_at, rss_bytes):
    result = parse_feed(rss_bytes, feed, retrieved_at)
    assert len(result.records) == 5
    assert sum(record.article is None for record in result.records) == 2
    assert json.loads(result.records[-2].raw_json)["entry"]["link"].endswith("missing-title")

    class Fetcher:
        def fetch(self, url):
            return b'<rss version="2.0"><channel><item><title>!!!</title><link>https://example.com</link></item></channel></rss>'

    collected = RSSCollector(Fetcher()).collect([feed])
    assert collected.skipped_entries == 1
    assert collected.articles == []
    assert collected.records[0].issues == ["no_meaningful_title"]


def test_snapshots_preserve_postgres_unrepresentable_input():
    raw = {"title": "a\x00b\ud800"}
    assert json.loads(snapshot(raw)) == raw
    assert "\x00" not in snapshot(raw)


def test_snapshot_keeps_script_markup_before_cleaning(feed, retrieved_at):
    content = b"""<rss version="2.0"><channel><item><title>Valid title</title>
    <description><![CDATA[<script>danger()</script><p>Visible content</p>]]></description>
    <link>https://example.com/a</link></item></channel></rss>"""
    record = parse_feed(content, feed, retrieved_at).records[0]
    assert "<script>" in json.loads(record.raw_json)["entry"]["summary"]
    assert record.article.description == "Visible content"
