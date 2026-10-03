from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from backend.app.schemas.chat import ChatResponse


class ConversationInfo(BaseModel):
    id: UUID
    revision: int
    created_at: datetime
    updated_at: datetime


class ConversationTurn(BaseModel):
    sequence: int
    question: str
    resolved_question: str
    response: ChatResponse
    created_at: datetime


class ConversationHistory(BaseModel):
    conversation: ConversationInfo
    turns: list[ConversationTurn]
    next_before: int | None = None
