import logging
from collections import Counter

from ingestion.cleaners.processor import ProcessingResult
from nlp.summarization.summarizer import ArticleSummarizer


def summarize_records(
    records: list[ProcessingResult], extractor: ArticleSummarizer | None = None
) -> None:
    extractor = extractor or ArticleSummarizer()
    for record in records:
        if record.article is not None:
            record.summary_result = extractor.summarize(
                record.article.title,
                record.article.description,
                record.content,
                record.language_result.language if record.language_result else None,
            )
    logging.getLogger(__name__).info(
        "article_summarization_complete outcomes=%s",
        dict(
            Counter(
                record.summary_result.summary_status
                for record in records
                if record.summary_result is not None
            )
        ),
    )
