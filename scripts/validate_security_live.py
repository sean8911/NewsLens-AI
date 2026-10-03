"""Read-only native HTTP acceptance using a temporary API process and real PostgreSQL."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from backend.app.core.config import Settings
from backend.app.core.version import VERSION
from backend.app.database.session import create_database_engine
from scripts.validate_multilingual_live import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local/security-validation")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(Settings())
    report = {"passed": False, "validated_at": datetime.now(UTC).isoformat(), "checks": []}
    report["before"] = fingerprint(engine)
    environment = os.environ.copy()
    environment.update(
        {
            "NEWSLENS_API_INFERENCE_REQUESTS_PER_MINUTE": "5",
            "NEWSLENS_API_REQUESTS_PER_MINUTE": "30",
            "NEWSLENS_API_MAX_BODY_BYTES": "16384",
        }
    )
    # Ask Windows for an unused local port. This process owns only the child it starts.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    child = None
    try:
        with (args.output_dir / "server.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                [sys.executable, "-m", "backend.app.api_cli", "--port", str(port)],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            with requests.Session() as http:
                http.trust_env = False
                for _ in range(100):
                    if child.poll() is not None:
                        raise RuntimeError("Temporary API exited before startup")
                    try:
                        health = http.get(base + "/health", timeout=1)
                        if health.status_code == 200:
                            break
                    except requests.ConnectionError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Temporary API startup timed out")
                assert health.json()["version"] == VERSION
                report["version"] = VERSION
                report["checks"].append("native_cli_health_and_version")
                status = http.get(base + "/system/status", timeout=10)
                assert status.status_code == 200 and status.json()["database"] == "available"
                report["checks"].append("real_postgresql_status")
                large = http.post(base + "/chat", json={"question": "x" * 17000}, timeout=5)
                assert large.status_code == 413
                report["checks"].append("oversized_body_rejected")
                invalid = http.post(
                    base + "/chat",
                    json={
                        "question": "private-sentinel",
                        "private-sentinel": "private-sentinel",
                    },
                    timeout=5,
                )
                assert invalid.status_code == 422 and "private-sentinel" not in invalid.text
                report["checks"].append("validation_redacts_rejected_input")
                for _ in range(3):
                    result = http.get(
                        base + "/search",
                        params={
                            "q": "NVIDIA",
                            "method": "lexical",
                            "recency": "off",
                        },
                        timeout=10,
                    )
                    assert result.status_code == 200 and result.json()["method"] == "lexical"
                report["checks"].append("real_lexical_search_within_budget")
                for headers in ({}, {"X-Forwarded-For": "198.51.100.1"}):
                    limited = http.get(base + "/search?q=NVIDIA", headers=headers, timeout=5)
                    assert limited.status_code == 429
                    assert 1 <= int(limited.headers["Retry-After"]) <= 60
                    assert limited.headers["Cache-Control"] == "no-store"
                    assert limited.headers["X-Content-Type-Options"] == "nosniff"
                report["checks"].append("rate_limit_and_forwarded_header_spoof_rejection")
                assert http.get(base + "/health", timeout=5).status_code == 200
                report["checks"].append("health_available_after_rate_limit")
        report["after"] = fingerprint(engine)
        assert report["after"] == report["before"]
        report["checks"].append("native_article_corpus_unchanged")
        report["passed"] = True
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        engine.dispose()
        (args.output_dir / "live.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
