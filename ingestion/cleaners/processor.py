import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from backend.app.schemas.article import Article
from ingestion.cleaners.dates import clean_publication_date
from ingestion.cleaners.sources import normalize_source
from ingestion.cleaners.text import clean_text, meaningful
from ingestion.cleaners.urls import clean_url
from nlp.classification.classifier import TopicResult
from nlp.entities.extractor import EntityResult
from nlp.keywords.extractor import KeywordResult
from nlp.language.detector import LanguageResult
from nlp.summarization.summarizer import SummaryResult

CLEANING_VERSION = "1"


def snapshot(data: dict[str, Any]) -> str:
    # Text containing ASCII-escaped JSON preserves even NUL and lone surrogate inputs,
    # which PostgreSQL JSONB cannot represent. Never silently drop those characters.
    return json.dumps(data, ensure_ascii=True, sort_keys=True, default=str)


@dataclass
class ProcessingResult:
    raw_json: str
    article: Article | None
    issues: list[str] = field(default_factory=list)
    content: str | None = None
    content_available: bool = False
    content_extraction_method: str = "none"
    content_extraction_status: str = "not_attempted"
    language_result: LanguageResult | None = None
    topic_result: TopicResult | None = None
    entity_result: EntityResult | None = None
    keyword_result: KeywordResult | None = None
    summary_result: SummaryResult | None = None


class ArticleProcessor:
    def process(
        self, article: Article, *, raw_json: str | None = None, content: str | None = None
    ) -> ProcessingResult:
        original = article.model_dump(mode="json")
        raw_json = raw_json or snapshot(
            {"format": "normalized_article", "article": original, "content": content}
        )
        cleaned = dict(original)
        issues = []
        for name in ("title", "description", "author", "category"):
            cleaned[name] = clean_text(original[name], single_line=name != "description")
            if cleaned[name] != original[name]:
                issues.append(f"{name}_cleaned")
        cleaned["source"] = normalize_source(article.source)
        if cleaned["source"] != article.source:
            issues.append("source_normalized")
        cleaned["rss_categories"] = list(
            dict.fromkeys(
                term
                for tag in article.rss_categories
                if (term := clean_text(tag, single_line=True))
            )
        )
        if cleaned["rss_categories"] != article.rss_categories:
            issues.append("rss_categories_cleaned")
        try:
            cleaned["url"] = clean_url(str(article.url))
        except ValueError:
            return ProcessingResult(raw_json, None, issues + ["invalid_url"])
        if cleaned["url"] != str(article.url):
            issues.append("url_tracking_removed")
        raw_entry = json.loads(raw_json).get("entry", {})
        published, date_issue = clean_publication_date(
            article.published_at, article.retrieved_at, raw_date=raw_entry.get("published")
        )
        cleaned["published_at"] = published
        if date_issue:
            issues.append(date_issue)
        if not meaningful(cleaned["title"]):
            return ProcessingResult(raw_json, None, issues + ["no_meaningful_title"])
        if cleaned["description"] == cleaned["title"]:
            cleaned["description"] = None
            issues.append("description_duplicates_title")
        elif cleaned["description"] and not meaningful(cleaned["description"]):
            cleaned["description"] = None
            issues.append("description_not_meaningful")
        if not cleaned["category"]:
            cleaned["category"] = "Other"
            issues.append("empty_category")
        clean_content = clean_text(content)
        if clean_content != content:
            issues.append("content_cleaned")
        try:
            result = Article.model_validate(cleaned)
        except ValidationError:
            return ProcessingResult(raw_json, None, issues + ["invalid_clean_record"])
        return ProcessingResult(raw_json, result, issues, clean_content)
