"""Short transactions; no database transaction spans retrieval or generation."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select, update

from backend.app.models.conversation import ConversationRecord, ConversationTurnRecord
from backend.app.schemas.conversation import (
    ConversationHistory,
    ConversationInfo,
    ConversationTurn,
)

MAX_TURNS = 200
HISTORY_WINDOW = 4


class ConversationError(Exception):
    status_code = 409


class ConversationFiltersInvalid(ConversationError):
    status_code = 422

    def __init__(self):
        super().__init__(
            "Invalid conversation filters. Supply a valid date range or reset context."
        )


class ConversationNotFound(ConversationError):
    status_code = 404

    def __init__(self):
        super().__init__("Conversation not found")


class ConversationConflict(ConversationError):
    def __init__(self):
        super().__init__("Conversation changed. Read its revision and retry.")


class ConversationCapacity(ConversationError):
    def __init__(self):
        super().__init__("Conversation reached 200 turns. Start a new conversation.")


def info(record):
    return ConversationInfo.model_validate(record, from_attributes=True)


class ConversationStore:
    def __init__(self, sessions):
        self.sessions = sessions

    def create(self):
        now = datetime.now(UTC)
        with self.sessions.begin() as session:
            record = ConversationRecord(
                id=uuid4(), revision=0, memory={}, created_at=now, updated_at=now
            )
            session.add(record)
            session.flush()
            return info(record)

    def snapshot(self, id_: UUID, expected_revision: int):
        with self.sessions.begin() as session:
            record = session.scalar(
                select(ConversationRecord).where(ConversationRecord.id == id_).with_for_update()
            )
            if record is None:
                raise ConversationNotFound()
            if record.revision != expected_revision:
                raise ConversationConflict()
            if record.revision >= MAX_TURNS:
                raise ConversationCapacity()
            turns = list(
                session.scalars(
                    select(ConversationTurnRecord)
                    .where(ConversationTurnRecord.conversation_id == id_)
                    .order_by(ConversationTurnRecord.sequence.desc())
                    .limit(HISTORY_WINDOW)
                )
            )
            return dict(record.memory), list(reversed(turns))

    def append(self, request, response, memory, filters):
        now = datetime.now(UTC)
        with self.sessions.begin() as session:
            changed = session.execute(
                update(ConversationRecord)
                .where(
                    ConversationRecord.id == request.conversation_id,
                    ConversationRecord.revision == request.expected_revision,
                    ConversationRecord.revision < MAX_TURNS,
                )
                .values(revision=request.expected_revision + 1, memory=memory, updated_at=now)
            )
            if changed.rowcount != 1:
                raise ConversationConflict()
            session.add(
                ConversationTurnRecord(
                    conversation_id=request.conversation_id,
                    sequence=request.expected_revision + 1,
                    question=request.question,
                    resolved_question=response.resolved_question,
                    filters=filters,
                    response=response.model_dump(mode="json"),
                    created_at=now,
                )
            )

    def history(self, id_: UUID, limit=20, before=None):
        if not 1 <= limit <= 50 or (before is not None and before < 1):
            raise ValueError("Invalid history page")
        with self.sessions.begin() as session:
            record = session.scalar(
                select(ConversationRecord).where(ConversationRecord.id == id_).with_for_update()
            )
            if record is None:
                raise ConversationNotFound()
            query = select(ConversationTurnRecord).where(
                ConversationTurnRecord.conversation_id == id_
            )
            if before is not None:
                query = query.where(ConversationTurnRecord.sequence < before)
            records = list(
                session.scalars(
                    query.order_by(ConversationTurnRecord.sequence.desc()).limit(limit + 1)
                )
            )
            page = records[:limit]
            return ConversationHistory(
                conversation=info(record),
                turns=[
                    ConversationTurn.model_validate(r, from_attributes=True) for r in reversed(page)
                ],
                next_before=page[-1].sequence if len(records) > limit else None,
            )
