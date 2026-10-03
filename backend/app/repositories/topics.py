from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend.app.models.article_topic import ArticleTopicRecord
from nlp.classification.classifier import TopicResult


class TopicRepository:
    def __init__(self, session: Session):
        self.session = session

    def replace(self, article_id: int, result: TopicResult) -> None:
        self.session.execute(
            delete(ArticleTopicRecord).where(ArticleTopicRecord.article_id == article_id)
        )
        self.session.add_all(
            [
                ArticleTopicRecord(
                    article_id=article_id, topic=label.topic, confidence=label.confidence
                )
                for label in result.article_topics
            ]
        )
