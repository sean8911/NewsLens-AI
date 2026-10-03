from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend.app.models.article_keyword import ArticleKeywordRecord
from nlp.keywords.extractor import KeywordResult


class KeywordRepository:
    def __init__(self, session: Session):
        self.session = session

    def replace(self, article_id: int, result: KeywordResult) -> None:
        self.session.execute(
            delete(ArticleKeywordRecord).where(ArticleKeywordRecord.article_id == article_id)
        )
        self.session.add_all(
            [
                ArticleKeywordRecord(article_id=article_id, **item)
                for item in result.values()["keywords"]
            ]
        )
