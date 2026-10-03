import logging
from collections import Counter

from ingestion.cleaners.processor import ProcessingResult
from nlp.language.detector import ArticleLanguageDetector

logger = logging.getLogger(__name__)


def detect_records(
    records: list[ProcessingResult], detector: ArticleLanguageDetector | None = None
) -> None:
    detector = detector or ArticleLanguageDetector()
    for record in records:
        if record.article is not None:
            record.language_result = detector.detect(
                record.article.title, record.article.description, record.content
            )
    logger.info(
        "language_detection_complete outcomes=%s",
        dict(
            Counter(
                record.language_result.language or record.language_result.language_status
                for record in records
                if record.language_result is not None
            )
        ),
    )
