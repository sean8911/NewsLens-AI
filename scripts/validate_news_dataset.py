"""Phase 26 acceptance: offline data, real CLI, read-only PostgreSQL and corpus integrity."""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine
from backend.app.evaluation.news_dataset import QUESTION_TYPES, load_dataset
from backend.app.evaluation.news_dataset_storage import compare_database

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(engine):
    with engine.connect() as connection:
        digest, count = hashlib.sha256(), 0
        for row in connection.scalars(
            text("SELECT row_to_json(a)::text FROM articles a ORDER BY id")
        ):
            digest.update(row.encode("utf-8") + b"\n")
            count += 1
        return {"count": count, "sha256": digest.hexdigest()}


def command(*args):
    result = subprocess.run(
        [sys.executable, "-m", "backend.app.evaluation.dataset_cli", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("Dataset CLI acceptance failed")
    return json.loads(result.stdout)


def main():
    settings = Settings()
    engine = create_database_engine(settings)
    export_path = ROOT / ".local" / f"phase26-questions-{uuid4().hex}.jsonl"
    report = {
        "passed": False,
        "validated_at": datetime.now(UTC).isoformat(),
        "mode": "offline-dataset-real-cli-read-only-postgresql",
        "before": fingerprint(engine),
    }
    try:
        dataset = load_dataset()
        report["dataset"] = dataset.summary()
        assert report["dataset"]["question_types"] == dict.fromkeys(QUESTION_TYPES, 10)
        report["database"] = compare_database(dataset, settings)
        assert report["database"]["matches"]
        report["cli_offline"] = command("validate")
        report["cli_database"] = command("validate", "--check-database")
        report["cli_export"] = command("export-questions", "--output", str(export_path))
        inputs = [json.loads(line) for line in export_path.read_text(encoding="utf-8").splitlines()]
        assert len(inputs) == 70
        assert inputs == [case.question_input() for case in dataset.questions]
        assert all(set(row) == {"id", "question", "filters", "time_context"} for row in inputs)
        report["label_free_export_verified"] = True
        report["passed"] = True
    finally:
        if export_path.exists():
            export_path.unlink()  # Only this invocation's unique output file.
        report["after"] = fingerprint(engine)
        report["articles_unchanged"] = report["before"] == report["after"]
        report["passed"] = report["passed"] and report["articles_unchanged"]
        engine.dispose()
        (ROOT / "docs/phase-26-live.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    assert report["passed"]
    print(
        json.dumps(
            {
                "passed": True,
                "questions": 70,
                "articles_unchanged": True,
                "retrieval_or_rag_metrics_measured": False,
            }
        )
    )


if __name__ == "__main__":
    main()
