import json
import logging
from collections import Counter

from backend.app.core.config import Settings
from ingestion.cleaners.processor import ProcessingResult
from ingestion.extraction.http import ArticleFetcher, ExtractionBlocked
from ingestion.extraction.parsers import rss_body, webpage_body

logger = logging.getLogger(__name__)


def extract_rss(record: ProcessingResult, minimum: int = 300) -> None:
    if record.article is None:
        return
    try:
        entry = json.loads(record.raw_json).get("entry", {})
        body = rss_body(entry, record.article.description, minimum)
        record.content_extraction_status = "rss_only"
        if body:
            record.content = body
            record.content_available = True
            record.content_extraction_method = "rss_content"
            record.content_extraction_status = "extracted"
    except Exception as exc:
        logger.warning("rss_content_failed error=%s", type(exc).__name__)
        record.content_extraction_status = "extraction_error"


def enrich_web(
    records: list[ProcessingResult],
    settings: Settings,
    fetcher: ArticleFetcher,
    *,
    already_available: set[str] | None = None,
) -> None:
    seen = set(already_available or ())
    seen.update(
        str(record.article.url)
        for record in records
        if record.article is not None and record.content_available
    )
    attempts = 0
    for record in records:
        if record.article is None or record.content_available:
            continue
        url = str(record.article.url)
        if url in seen:
            record.content_extraction_status = "already_seen"
            continue
        seen.add(url)
        if not settings.content_web_enabled:
            record.content_extraction_status = "web_disabled"
            continue
        try:
            # Validate policy before consuming the per-run page budget.
            _, policy = fetcher.validate(url)
            if attempts >= settings.content_max_pages_per_run:
                record.content_extraction_status = "budget_exhausted"
                continue
            attempts += 1
            html = fetcher.fetch(url)
            body, method, status = webpage_body(
                html,
                record.article.description,
                selector=policy.article_selector,
                minimum=settings.content_min_chars,
            )
            record.content_extraction_status = status
            if body:
                record.content = body
                record.content_available = True
                record.content_extraction_method = method
        except ExtractionBlocked as exc:
            record.content_extraction_status = exc.reason
        except Exception as exc:
            # Each webpage is an independent failure boundary; keep the RSS article usable.
            record.content_extraction_status = "extraction_error"
            logger.warning("article_extraction_failed error=%s", type(exc).__name__)
    logger.info(
        "content_extraction_complete outcomes=%s",
        dict(
            Counter(
                record.content_extraction_status for record in records if record.article is not None
            )
        ),
    )
