import signal
from datetime import UTC, datetime, timedelta
from threading import Event
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from ingestion.scheduler import JOB_ID, build_scheduler, main, serve


def test_defaults_and_single_job_policy():
    settings = Settings(_env_file=None)
    scheduler = build_scheduler(settings, lambda: None)
    assert not scheduler.running  # Building/importing never launches threads.
    job = scheduler.get_job(JOB_ID)
    assert job.trigger.interval == timedelta(minutes=10)
    assert job.max_instances == 1
    assert job.coalesce is True
    assert job.misfire_grace_time == 60
    assert job.next_run_time <= datetime.now(UTC)


def test_delayed_start_uses_interval():
    settings = Settings(_env_file=None, scheduler_run_immediately=False)
    scheduler = build_scheduler(settings, lambda: None)
    scheduler.start(paused=True)
    try:
        remaining = scheduler.get_job(JOB_ID).next_run_time - datetime.now(UTC)
        assert timedelta(minutes=9) < remaining <= timedelta(minutes=10)
    finally:
        scheduler.shutdown()


def test_disabled_scheduler_performs_no_work(monkeypatch, caplog):
    monkeypatch.setenv("NEWSLENS_SCHEDULER_ENABLED", "false")
    import ingestion.scheduler as module

    def fail(*args):
        pytest.fail("Disabled scheduling must not create a database connection")

    monkeypatch.setattr(module, "create_database_engine", fail)
    assert build_scheduler(Settings(), lambda: None) is None
    assert main(["run"]) == 0


@pytest.mark.parametrize("value", [0, -1, 86401])
def test_invalid_interval_rejected(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, scheduler_interval_seconds=value)


def test_shutdown_waits_and_restores_signal_handlers():
    scheduler = MagicMock()
    scheduler.running = True
    stop = Event()
    stop.set()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    serve(scheduler, stop)
    scheduler.start.assert_called_once()
    scheduler.shutdown.assert_called_once_with(wait=True)
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_slow_job_does_not_overlap_next_interval():
    from apscheduler.events import EVENT_JOB_MAX_INSTANCES

    entered, release, skipped = Event(), Event(), Event()
    calls = []

    def slow_job():
        calls.append(1)
        entered.set()
        release.wait(10)

    scheduler = build_scheduler(Settings(_env_file=None, scheduler_interval_seconds=1), slow_job)
    scheduler.add_listener(lambda _: skipped.set(), EVENT_JOB_MAX_INSTANCES)
    scheduler.start()
    try:
        assert entered.wait(5)
        assert skipped.wait(5)
    finally:
        release.set()
        scheduler.shutdown(wait=True)
    assert len(calls) == 1
