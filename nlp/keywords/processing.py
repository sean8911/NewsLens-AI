import logging
from collections import Counter

from ingestion.cleaners.processor import ProcessingResult
from nlp.keywords.extractor import KeywordExtractor


def extract_records(
    records: list[ProcessingResult], extractor: KeywordExtractor | None = None
) -> None:
    extractor = extractor or KeywordExtractor()
    for record in records:
        if record.article is not None:
            record.keyword_result = extractor.extract(
                record.article.title,
                record.article.description,
                record.content,
                record.language_result.language if record.language_result else None,
            )
    logging.getLogger(__name__).info(
        "keyword_extraction_complete outcomes=%s",
        dict(
            Counter(
                record.keyword_result.keyword_status
                for record in records
                if record.keyword_result is not None
            )
        ),
    )
