from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.ingestion_run import IngestionRun
from backend.app.schemas.ingestion_run import RunReport, UpdateStatus


class IngestionRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def start(self, *, started_at: datetime, feeds_checked: int) -> int:
        run = IngestionRun(started_at=started_at, feeds_checked=feeds_checked)
        self.session.add(run)
        self.session.flush()
        return run.id

    def finish(self, run_id: int, **values: object) -> RunReport:
        run = self.session.scalar(
            select(IngestionRun).where(IngestionRun.id == run_id).with_for_update()
        )
        if run is None:
            raise LookupError("Update run no longer exists")
        # Completion is immutable. If the client lost a commit acknowledgement, a
        # failure-recovery attempt must not overwrite an already committed success.
        if run.status != "running":
            return RunReport.model_validate(run)
        for key, value in values.items():
            setattr(run, key, value)
        self.session.flush()
        return RunReport.model_validate(run)

    def status(self, limit: int = 5) -> UpdateStatus:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        # Derive freshness from committed outcomes; failures cannot erase the last good update.
        successful = self.session.scalar(
            select(func.max(IngestionRun.finished_at)).where(
                IngestionRun.status.in_(["succeeded", "partial"])
            )
        )
        complete = self.session.scalar(
            select(func.max(IngestionRun.finished_at)).where(IngestionRun.status == "succeeded")
        )
        runs = self.session.scalars(
            select(IngestionRun).order_by(IngestionRun.id.desc()).limit(limit)
        )
        return UpdateStatus(
            last_successful_news_update=successful,
            last_complete_news_update=complete,
            recent_runs=[RunReport.model_validate(run) for run in runs],
        )
