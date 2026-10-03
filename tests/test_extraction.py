import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import requests
from pydantic import ValidationError

from backend.app.core.config import Settings
from ingestion.extraction import http
from ingestion.extraction.http import ArticleFetcher, ExtractionBlocked
from ingestion.extraction.parsers import webpage_body
from ingestion.extraction.policy import SourcePolicy, load_policies
from ingestion.extraction.service import enrich_web
from ingestion.parsers.rss import parse_feed

BODY = (
    "Researchers in Malaysia published the results of a new technology study on Wednesday. "
    "The team compared several methods using independently collected data and documented "
    "their findings for other researchers to reproduce. Further work will examine how the "
    "results apply across different communities and operating conditions."
)
HTML = (
    "<article><p>" + BODY + "</p><p>A second paragraph with further context.</p></article>"
).encode()


def policy(**changes):
    return SourcePolicy.model_validate(
        {
            "host": "example.com",
            "terms_allow_extraction": True,
            "terms_url": "https://example.com/terms",
            "review_note": "Synthetic test permission",
            **changes,
        }
    )


class Response:
    def __init__(self, body=b"", status=200, mime="text/html", **headers):
        self.body, self.status_code = body, status
        self.headers = {"Content-Type": mime, **headers}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, _):
        yield self.body


@pytest.fixture
def client(monkeypatch):
    http._NEXT_REQUEST.clear()
    clock = [100.0]
    monkeypatch.setattr(http.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        http.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(
        http.socket, "getaddrinfo", lambda *_a, **_kw: [(2, 1, 6, "", ("93.184.216.34", 443))]
    )
    fetcher = ArticleFetcher(Settings(_env_file=None), {"example.com": policy()})
    calls = []
    routes = {
        "https://example.com/robots.txt": Response(
            b"User-agent: *\nAllow: /\nCrawl-delay: 3\n", mime="text/plain"
        ),
        "https://example.com/story": Response(HTML),
    }

    def get(url, **kwargs):
        assert kwargs["allow_redirects"] is False
        assert kwargs["stream"] is True
        calls.append((url, clock[0]))
        response = routes[url]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(fetcher.session, "get", get)
    yield SimpleNamespace(fetcher=fetcher, routes=routes, calls=calls, clock=clock)
    fetcher.close()
    http._NEXT_REQUEST.clear()


def rss_with_content(body=BODY):
    return (
        f'<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        f"<channel><title>Example</title><item><title>Study results</title>"
        f"<link>https://example.com/story</link><description>Short RSS summary.</description>"
        f"<content:encoded><![CDATA[<p>{body}</p>]]></content:encoded>"
        f"</item></channel></rss>"
    ).encode()


def test_rss_body_is_clean_and_raw_is_preserved(feed, retrieved_at):
    record = parse_feed(rss_with_content(), feed, retrieved_at).records[0]
    assert record.content == BODY
    assert record.content_available and record.content_extraction_method == "rss_content"
    assert record.article.description == "Short RSS summary."
    assert "<p>" in record.raw_json


def test_atom_content_without_summary_is_retained(feed, retrieved_at):
    atom = (
        f'<feed xmlns="http://www.w3.org/2005/Atom"><title>Example</title><entry>'
        f'<title>Study</title><link href="https://example.com/story"/>'
        f'<content type="text">{BODY}</content></entry></feed>'
    ).encode()
    record = parse_feed(atom, feed, retrieved_at).records[0]
    assert record.content == BODY
    assert record.content_available


def test_short_feed_content_remains_description_only(feed, retrieved_at):
    record = parse_feed(rss_with_content("Too short."), feed, retrieved_at).records[0]
    assert record.content is None and not record.content_available
    assert record.article.description == "Short RSS summary."


def test_jsonld_graph_precedes_html_and_skips_malformed_json():
    metadata = {"@graph": [{"@type": ["NewsArticle"], "articleBody": BODY}]}
    html = (
        b'<script type="application/ld+json">bad JSON</script>'
        + ('<script type="application/ld+json">' + json.dumps(metadata) + "</script>").encode()
        + HTML
    )
    assert webpage_body(html, "Summary") == (BODY, "json_ld", "extracted")


def test_source_selector_removes_navigation_and_hidden_content():
    html = (
        f'<nav>{BODY}</nav><div class="story"><p>{BODY}</p>'
        "<p hidden>Do not extract this.</p><p>More useful context.</p>"
        "<aside><p>Related stories.</p></aside></div>"
    ).encode()
    body, method, _ = webpage_body(html, None, selector=".story")
    assert method == "html_article"
    assert "More useful" in body
    assert "Do not extract" not in body and "Related stories" not in body
    assert webpage_body(html, None)[0] is None  # No broad whole-page fallback.


@pytest.mark.parametrize(
    "marker",
    [
        '<script type="application/ld+json">{"isAccessibleForFree":false}</script>',
        '<div class="paywall">Subscribe to continue</div>',
        '<form id="challenge-form">Verify</form>',
    ],
)
def test_restricted_pages_never_yield_hidden_article_body(marker):
    assert webpage_body(marker.encode() + HTML, None) == (None, "none", "restricted")


def test_summary_metadata_is_not_full_content():
    html = ('<meta property="og:description" content="' + BODY + '">').encode()
    assert webpage_body(html, "Summary")[0] is None


def test_robots_checked_cached_and_delay_applied(client):
    assert client.fetcher.fetch("https://example.com/story") == HTML
    assert client.fetcher.fetch("https://example.com/story") == HTML
    assert [time for _, time in client.calls] == [100, 103, 106]
    assert len([url for url, _ in client.calls if url.endswith("robots.txt")]) == 1


def test_robots_wildcards_and_specific_allow(client):
    client.routes["https://example.com/robots.txt"] = Response(
        b"User-agent: *\nDisallow: /story*\nAllow: /story/public$\n", mime="text/plain"
    )
    with pytest.raises(ExtractionBlocked, match="robots_denied"):
        client.fetcher.fetch("https://example.com/story")
    assert len(client.calls) == 1
    client.routes["https://example.com/story/public"] = Response(HTML)
    assert client.fetcher.fetch("https://example.com/story/public") == HTML


@pytest.mark.parametrize(
    "response",
    [
        Response(status=404, mime="text/plain"),
        Response(status=302, Location="/robots-new.txt"),
        Response(b"<html>Error</html>"),
        Response(b"Upstream error", mime="text/plain"),
        requests.Timeout("private message"),
    ],
)
def test_unavailable_robots_fails_closed(client, response):
    client.routes["https://example.com/robots.txt"] = response
    with pytest.raises((ExtractionBlocked, requests.Timeout)):
        client.fetcher.fetch("https://example.com/story")
    with pytest.raises(ExtractionBlocked, match="robots_unavailable"):
        client.fetcher.fetch("https://example.com/story")
    assert len(client.calls) == 1


def test_redirect_target_is_checked_against_robots(client):
    client.routes["https://example.com/robots.txt"] = Response(
        b"User-agent: *\nDisallow: /private\n", mime="text/plain"
    )
    client.routes["https://example.com/story"] = Response(status=302, Location="/private")
    with pytest.raises(ExtractionBlocked, match="robots_denied"):
        client.fetcher.fetch("https://example.com/story")
    assert len(client.calls) == 2


def test_cross_host_redirect_is_not_requested(client):
    client.routes["https://example.com/story"] = Response(
        status=302, Location="https://other.com/a"
    )
    with pytest.raises(ExtractionBlocked, match="redirect_host_denied"):
        client.fetcher.fetch("https://example.com/story")
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/story",
        "https://name:password@example.com/story",
        "https://example.com:444/story",
        "https://unapproved.com/story",
    ],
)
def test_unapproved_or_unsafe_target_never_requested(client, url):
    with pytest.raises(ExtractionBlocked):
        client.fetcher.fetch(url)
    assert not client.calls


def test_private_dns_is_rejected(client, monkeypatch):
    monkeypatch.setattr(
        http.socket, "getaddrinfo", lambda *_a, **_kw: [(2, 1, 6, "", ("127.0.0.1", 443))]
    )
    with pytest.raises(ExtractionBlocked, match="unsafe_url"):
        client.fetcher.fetch("https://example.com/story")
    assert not client.calls


def test_retry_after_persists_across_fetcher_instances(client):
    client.routes["https://example.com/story"] = Response(status=429, **{"Retry-After": "7200"})
    with pytest.raises(ExtractionBlocked, match="http_restricted"):
        client.fetcher.fetch("https://example.com/story")
    other = ArticleFetcher(Settings(_env_file=None), client.fetcher.policies)
    try:
        with pytest.raises(ExtractionBlocked, match="rate_deferred"):
            other.fetch("https://example.com/story")
        assert http._NEXT_REQUEST["example.com"] >= client.clock[0] + 7200
    finally:
        other.close()


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response(b"x" * 2048), "too_large"),
        (Response(HTML, mime="application/pdf"), "unsupported_type"),
    ],
)
def test_bounded_html_only_download(client, response, reason):
    client.fetcher.settings.content_max_bytes = 1024
    client.routes["https://example.com/story"] = response
    with pytest.raises(ExtractionBlocked, match=reason):
        client.fetcher.fetch("https://example.com/story")


def test_rate_directive_respected_and_excessive_delay_skipped(client):
    client.routes["https://example.com/robots.txt"] = Response(
        b"User-agent: *\nRequest-rate: 1/10s\n", mime="text/plain"
    )
    client.fetcher.fetch("https://example.com/story")
    assert client.calls[1][1] - client.calls[0][1] >= 10
    client.fetcher.robots.clear()
    client.routes["https://example.com/robots.txt"] = Response(
        b"User-agent: *\nCrawl-delay: 120\n", mime="text/plain"
    )
    with pytest.raises(ExtractionBlocked, match="crawl_delay_exceeded"):
        client.fetcher.fetch("https://example.com/story")


def test_failed_extractor_preserves_summary_and_next_article_runs(
    client, feed, retrieved_at, caplog
):
    first = parse_feed(rss_with_content("short"), feed, retrieved_at).records[0]
    second = deepcopy(first)
    second.article = second.article.model_copy(update={"url": "https://example.com/good"})
    client.routes["https://example.com/story"] = requests.Timeout("sensitive URL")
    client.routes["https://example.com/good"] = Response(HTML)
    enrich_web([first, second], Settings(_env_file=None, content_web_enabled=True), client.fetcher)
    assert first.content_extraction_status == "extraction_error"
    assert first.article.description == "Short RSS summary."
    assert first.content is None
    assert second.content_available
    assert "sensitive URL" not in caplog.text


def test_rss_priority_disabled_web_duplicate_and_budget(client, feed, retrieved_at):
    full = parse_feed(rss_with_content(), feed, retrieved_at).records[0]
    missing = parse_feed(rss_with_content("short"), feed, retrieved_at).records[0]
    enrich_web([full, missing], Settings(_env_file=None), client.fetcher)
    assert not client.calls and full.content_available
    assert missing.content_extraction_status == "already_seen"
    enrich_web([missing], Settings(_env_file=None), client.fetcher)
    assert missing.content_extraction_status == "web_disabled"
    records = [deepcopy(missing) for _ in range(3)]
    records[-1].article = records[-1].article.model_copy(update={"url": "https://example.com/next"})
    settings = Settings(_env_file=None, content_web_enabled=True, content_max_pages_per_run=1)
    enrich_web(records, settings, client.fetcher)
    assert [record.content_extraction_status for record in records] == [
        "extracted",
        "already_seen",
        "budget_exhausted",
    ]


def test_reviewed_source_policy_is_required_and_validated(tmp_path, client):
    client.fetcher.policies = {"example.com": policy(terms_allow_extraction=False)}
    with pytest.raises(ExtractionBlocked, match="policy_denied"):
        client.fetcher.fetch("https://example.com/story")
    assert not client.calls
    with pytest.raises(ValidationError):
        policy(article_selector="[")
    path = tmp_path / "policies.json"
    path.write_text(json.dumps([policy().model_dump()] * 2))
    with pytest.raises(ValueError, match="Duplicate"):
        load_policies(path)
