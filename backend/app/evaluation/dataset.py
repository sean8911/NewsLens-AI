from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.search import SearchRequest


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=10000)
    description: str | None = Field(default=None, max_length=50000)
    content: str | None = Field(default=None, max_length=100000)


class QueryCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    q: str
    relevance: dict[int, Annotated[int, Field(strict=True, ge=0, le=3)]]

    @model_validator(mode="after")
    def valid_query(self) -> Self:
        SearchRequest(q=self.q)
        if not any(self.relevance.values()):
            raise ValueError("Each query needs a positive relevance judgment")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    note: str
    documents: list[Document] = Field(min_length=1, max_length=500)
    queries: list[QueryCase] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def valid_ids(self) -> Self:
        ids = {document.id for document in self.documents}
        if len(ids) != len(self.documents) or len({q.id for q in self.queries}) != len(
            self.queries
        ):
            raise ValueError("Document and query IDs must be unique")
        if any(set(query.relevance) - ids for query in self.queries):
            raise ValueError("Judgments must reference existing documents")
        return self
