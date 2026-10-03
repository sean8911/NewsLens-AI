import json
from collections.abc import Iterator
from typing import Any

from bs4 import BeautifulSoup

from ingestion.cleaners.text import clean_text, meaningful


def usable_body(value: Any, description: str | None, minimum: int = 300) -> str | None:
    if not isinstance(value, str):
        return None
    body = clean_text(value)
    summary = clean_text(description)
    # A long summary alone is not evidence of full article content.
    if not body or len(body) < minimum or not meaningful(body) or body == summary:
        return None
    return body


def rss_body(entry: dict, description: str | None, minimum: int = 300) -> str | None:
    candidates = entry.get("content", [])
    if not isinstance(candidates, list):
        return None
    bodies = [
        body
        for part in candidates
        if isinstance(part, dict)
        and part.get("type", "text/html") in {"text/html", "text/plain", "application/xhtml+xml"}
        and (body := usable_body(part.get("value"), None, minimum))
    ]
    return max(bodies, key=len, default=None)


def objects(value: Any, depth: int = 0) -> Iterator[dict]:
    if depth > 20:
        return
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from objects(nested, depth + 1)
    elif isinstance(value, list):
        for nested in value:
            yield from objects(nested, depth + 1)


def webpage_body(
    html: bytes, description: str | None, *, selector: str = "article", minimum: int = 300
) -> tuple[str | None, str, str]:
    soup = BeautifulSoup(html, "html.parser")
    metadata = []
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            metadata.extend(objects(json.loads(script.string or "")))
        except (ValueError, RecursionError):
            continue
    # Never extract text hidden behind an explicitly declared subscription barrier.
    if any(str(item.get("isAccessibleForFree", "")).lower() == "false" for item in metadata):
        return None, "none", "restricted"
    if soup.select_one('[class*="paywall"], [id*="paywall"], [class*="captcha"], #challenge-form'):
        return None, "none", "restricted"
    for item in metadata:
        types = item.get("@type", [])
        types = [types] if isinstance(types, str) else types
        if not isinstance(types, list) or not any(
            kind in {"Article", "NewsArticle", "ReportageNewsArticle", "AnalysisNewsArticle"}
            for kind in types
            if isinstance(kind, str)
        ):
            continue
        if body := usable_body(item.get("articleBody"), description, minimum):
            return body, "json_ld", "extracted"
    for unwanted in soup.select(
        "script, style, nav, header, footer, aside, form, button, noscript, "
        "[hidden], [aria-hidden='true']"
    ):
        unwanted.decompose()
    candidates = []
    for region in soup.select(selector):
        # Explicit article regions only; never use the entire page as a fallback.
        paragraphs = region.select("p")
        if len(paragraphs) >= 2:
            text = "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs)
            if body := usable_body(text, description, minimum):
                candidates.append(body)
    if candidates:
        return max(candidates, key=len), "html_article", "extracted"
    return None, "none", "no_content"
