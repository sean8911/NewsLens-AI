import logging
from collections import Counter

from ingestion.cleaners.processor import ProcessingResult
from nlp.classification.classifier import TopicClassifier


def classify_records(
    records: list[ProcessingResult], classifier: TopicClassifier | None = None
) -> None:
    classifier = classifier or TopicClassifier()
    for record in records:
        if record.article is not None:
            record.topic_result = classifier.classify(
                record.article.title,
                record.article.description,
                record.content,
                record.language_result.language if record.language_result else None,
            )
    logging.getLogger(__name__).info(
        "topic_classification_complete outcomes=%s",
        dict(
            Counter(
                record.topic_result.topic_status
                for record in records
                if record.topic_result is not None
            )
        ),
    )
