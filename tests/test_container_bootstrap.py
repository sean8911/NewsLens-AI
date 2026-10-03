from unittest.mock import Mock

import pytest
from sqlalchemy.engine import make_url

from infra.docker import bootstrap, entrypoint


def test_container_password_is_encoded_without_changing_database_scope():
    password = "p@ss:/?#% with spaces"
    url = make_url(entrypoint.database_url(password))
    assert url.password == password
    assert (url.host, url.port, url.database, url.username) == (
        "postgres",
        5432,
        "newslens",
        "newslens",
    )


def test_missing_container_password_stops_before_exec(monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    execute = Mock()
    monkeypatch.setattr(entrypoint.os, "execvp", execute)
    with pytest.raises(SystemExit, match="POSTGRES_PASSWORD"):
        entrypoint.main()
    execute.assert_not_called()


def test_entrypoint_overrides_host_database_and_executes_argument_vector(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "container-only")
    monkeypatch.setenv("NEWSLENS_DATABASE_URL", "postgresql://native.invalid/private")
    monkeypatch.setattr(
        entrypoint.sys, "argv", ["entrypoint.py", "python", "-m", "alembic", "upgrade", "head"]
    )
    execute = Mock()
    monkeypatch.setattr(entrypoint.os, "execvp", execute)
    entrypoint.main()
    execute.assert_called_once_with("python", ["python", "-m", "alembic", "upgrade", "head"])
    assert make_url(entrypoint.os.environ["NEWSLENS_DATABASE_URL"]).host == "postgres"


def test_qdrant_readiness_retries_then_allows_migrations(monkeypatch):
    response = Mock(status=200)
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    probe = Mock(side_effect=[OSError("not ready"), context])
    monkeypatch.setattr(bootstrap, "urlopen", probe)
    monkeypatch.setattr(bootstrap.time, "sleep", Mock())
    run = Mock()
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    bootstrap.main()
    assert probe.call_count == 2
    run.assert_called_once_with(
        [bootstrap.sys.executable, "-m", "alembic", "upgrade", "head"], check=True
    )


def test_qdrant_failure_is_bounded_and_does_not_run_migrations(monkeypatch):
    monkeypatch.setattr(bootstrap, "urlopen", Mock(side_effect=OSError("unavailable")))
    monkeypatch.setattr(bootstrap.time, "sleep", Mock())
    run = Mock()
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="did not become ready"):
        bootstrap.wait_for_qdrant(attempts=2)
    assert bootstrap.urlopen.call_count == 2
    run.assert_not_called()
