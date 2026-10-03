import json
import subprocess
from pathlib import Path

import pytest

from scripts import run_tests


@pytest.mark.parametrize(
    ("body", "exit_code", "status"),
    [
        ("<testcase/>", 0, "passed"),
        ("<testcase><skipped/></testcase>", 0, "failed"),
        ("<testcase/><testcase><skipped/></testcase>", 0, "failed"),
        ("<testcase><failure/></testcase>", 1, "failed"),
        ("<testcase><error/></testcase>", 1, "failed"),
        ("", 0, "failed"),
        ("<testcase/>", 2, "failed"),
        ("malformed", 0, "failed"),
        (None, 0, "failed"),
    ],
)
def test_acceptance_rejects_incomplete_runs(tmp_path, monkeypatch, body, exit_code, status):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k accidentally_narrow_selection")

    def execute(command, **kwargs):
        assert command[command.index("-m", 3) + 1] == "not integration"
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        assert kwargs["cwd"] == tmp_path
        report = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junit")))
        if body is not None:
            report.write_text(
                body
                if body == "malformed"
                else (f"<testsuites><testsuite>{body}</testsuite></testsuites>"),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, exit_code)

    monkeypatch.setattr(run_tests.subprocess, "run", execute)
    result = run_tests.run_suite("offline", tmp_path, tmp_path)
    assert result["status"] == status


def test_missing_database_is_failure_without_production_fallback(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("NEWSLENS_DATABASE_URL=private-production-url\n")

    def forbidden(*args, **kwargs):
        pytest.fail("No subprocess should run without a test database setting")

    monkeypatch.setattr(run_tests.subprocess, "run", forbidden)
    result = run_tests.run_suite("integration", tmp_path, tmp_path)
    assert result["status"] == "failed"
    assert result["reason"] == "missing_test_database_url"
    assert "private-production-url" not in (tmp_path / "integration.log").read_text()


def test_environment_or_dotenv_test_database_is_recognized(tmp_path, monkeypatch):
    assert not run_tests.database_configured(tmp_path)
    (tmp_path / ".env").write_text("NEWSLENS_TEST_DATABASE_URL=local-test-placeholder\n")
    assert run_tests.database_configured(tmp_path)
    monkeypatch.setenv("NEWSLENS_TEST_DATABASE_URL", "environment-test-placeholder")
    assert run_tests.database_configured(tmp_path)


def test_all_suites_run_and_failure_survives_in_unique_reports(tmp_path, monkeypatch):
    calls = []

    def suite(name, root, output):
        calls.append(name)
        return {"suite": name, "status": "failed" if name == "offline" else "passed"}

    monkeypatch.setattr(run_tests, "run_suite", suite)
    for _ in range(2):
        assert run_tests.main(["--output-dir", str(tmp_path)]) == 1
    assert calls == ["offline", "integration"] * 2
    reports = list(tmp_path.glob("pytest-*/summary.json"))
    assert len(reports) == 2
    assert all(json.loads(path.read_text())["status"] == "failed" for path in reports)
