"""Run the Phase 1-17 acceptance gate; never advances to the next phase.

Requires the configured application and dedicated test databases, the existing
article index, downloaded models, and network access to at least five RSS feeds.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine

ROOT = Path(__file__).resolve().parents[1]


def article_fingerprint() -> dict:
    engine = create_database_engine(Settings(_env_file=ROOT / ".env"))
    try:
        with engine.connect() as connection:
            rows = connection.scalars(
                text("SELECT row_to_json(a)::text FROM articles a ORDER BY id")
            )
            digest, count = hashlib.sha256(), 0
            for row in rows:
                digest.update(row.encode("utf-8") + b"\n")
                count += 1
            return {"count": count, "sha256": digest.hexdigest()}
    finally:
        engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/acceptance-1-17")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def artifact(name):
        return str(output / (name + ".json"))

    steps = [
        ("tests", ["-m", "pytest", "-q"]),
        ("lint", ["-m", "ruff", "check", "."]),
        ("format", ["-m", "ruff", "format", "--check", "."]),
        ("migrations", ["-m", "alembic", "check"]),
        ("dependencies", ["-m", "pip", "check"]),
        ("pipeline", ["scripts/validate_pipeline.py", "--output", artifact("pipeline")]),
        ("embeddings", ["scripts/validate_embeddings.py", "--output", artifact("embeddings")]),
        ("semantic", ["scripts/validate_semantic_search.py", "--output", artifact("semantic")]),
        ("lexical", ["scripts/validate_lexical_search.py", "--output", artifact("lexical")]),
        (
            "retrieval-evaluation",
            [
                "-m",
                "backend.app.evaluation.cli",
                "--compare-reranking",
                "--output",
                artifact("retrieval-evaluation"),
            ],
        ),
        (
            "temporal-evaluation",
            [
                "-m",
                "backend.app.evaluation.temporal_cli",
                "--output",
                artifact("temporal-evaluation"),
            ],
        ),
        (
            "chat-model",
            ["scripts/validate_chat.py", "--model-only", "--output", artifact("chat-model")],
        ),
        ("chat-http", ["scripts/validate_chat.py", "--output", artifact("chat-http")]),
        ("recency-http", ["scripts/validate_recency.py", "--output", artifact("recency-http")]),
    ]
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "articles_before": article_fingerprint(),
        "steps": [],
    }
    report_path = output / "result.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", "utf-8")

    save()
    try:
        for name, arguments in steps:
            print(f"Running {name}", flush=True)
            start = time.perf_counter()
            with (output / (name + ".log")).open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    [sys.executable, *arguments],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            report["steps"].append(
                {
                    "name": name,
                    "command": [sys.executable, *arguments],
                    "exit_code": process.returncode,
                    "seconds": round(time.perf_counter() - start, 2),
                }
            )
            save()
            if process.returncode:
                raise RuntimeError(f"{name} failed; see {name}.log")
            print(f"Passed {name}", flush=True)
        retrieval = json.loads(Path(artifact("retrieval-evaluation")).read_text("utf-8"))
        assert retrieval["hybrid_ndcg_wins_over_both"]["hybrid"], "Hybrid had no diagnostic win"
        report["retrieval_mean_metrics"] = retrieval["mean_metrics"]
        report["articles_after"] = article_fingerprint()
        assert report["articles_before"] == report["articles_after"], "Stored articles changed"
        report["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        print(f"Acceptance failed ({type(exc).__name__}); inspect {output}", flush=True)
        return 1
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        save()
        print(f"Acceptance {report['status']}: {report_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
