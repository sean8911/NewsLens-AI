import logging
from collections import Counter

from ingestion.cleaners.processor import ProcessingResult
from nlp.entities.extractor import EntityExtractor


def extract_records(
    records: list[ProcessingResult], extractor: EntityExtractor | None = None
) -> None:
    extractor = extractor or EntityExtractor()
    for record in records:
        if record.article is not None:
            record.entity_result = extractor.extract(
                record.article.title,
                record.article.description,
                record.content,
                record.language_result.language if record.language_result else None,
            )
    logging.getLogger(__name__).info(
        "entity_extraction_complete outcomes=%s",
        dict(
            Counter(
                record.entity_result.entity_status
                for record in records
                if record.entity_result is not None
            )
        ),
    )
