"""Run explicit pytest suites and reject skipped or empty acceptance runs."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def database_configured(root: Path) -> bool:
    return bool(
        os.environ.get("NEWSLENS_TEST_DATABASE_URL")
        or dotenv_values(root / ".env").get("NEWSLENS_TEST_DATABASE_URL")
    )


def junit_counts(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    cases = list(root.iter("testcase"))
    counts = {"tests": len(cases), "passed": 0, "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        if case.find("error") is not None:
            counts["errors"] += 1
        elif case.find("failure") is not None:
            counts["failures"] += 1
        elif case.find("skipped") is not None:
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    return counts


def run_suite(suite: str, root: Path, output: Path) -> dict:
    report = output / f"{suite}.xml"
    log = output / f"{suite}.log"
    result = {"suite": suite, "status": "failed", "junit": report.name, "log": log.name}
    if suite == "integration" and not database_configured(root):
        result["reason"] = "missing_test_database_url"
        log.write_text(
            "Set NEWSLENS_TEST_DATABASE_URL in the environment or repository .env.\n"
            "The application database URL is never used as a fallback.\n",
            encoding="utf-8",
        )
        return result
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "--tb=short",
        "--strict-markers",
        "--strict-config",
        "-o",
        "addopts=",
        "-m",
        "integration" if suite == "integration" else "not integration",
        f"--junitxml={report}",
    ]
    # User-supplied selection options must not silently narrow this acceptance run.
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTHONIOENCODING"] = "utf-8"
    started = perf_counter()
    print(f"Running {suite}; log: {log}", flush=True)
    try:
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result["exit_code"] = process.returncode
        result["counts"] = counts = junit_counts(report)
        if (
            process.returncode == 0
            and counts["passed"] > 0
            and not any(counts[key] for key in ("failures", "errors", "skipped"))
        ):
            result["status"] = "passed"
        else:
            result["reason"] = "pytest_failed_or_incomplete"
    except (OSError, ElementTree.ParseError):
        result["reason"] = "execution_or_report_error"
    result["seconds"] = round(perf_counter() - started, 3)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("offline", "integration", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local" / "test-results")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Each attempt gets its own directory; stale XML can never turn a crash into a pass.
    output = Path(tempfile.mkdtemp(prefix="pytest-", dir=args.output_dir.resolve()))
    suites = ("offline", "integration") if args.suite == "all" else (args.suite,)
    results = [run_suite(suite, ROOT, output) for suite in suites]
    passed = all(result["status"] == "passed" for result in results)
    summary = {
        "requested_suite": args.suite,
        "status": "passed" if passed else "failed",
        "completed_at": datetime.now(UTC).isoformat(),
        "suites": results,
        "scope": "Deterministic pytest regression tests; not live neural-model quality or Docker.",
    }
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
