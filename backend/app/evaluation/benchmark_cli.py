import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.evaluation.benchmark_runtime import run_benchmark
from backend.app.evaluation.news_dataset import DEFAULT_DATASET, load_dataset
from backend.app.evaluation.retrieval_benchmark import markdown_table, select_cases

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare four retrieval methods on the frozen dataset"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON report; never overwrite"
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--case", dest="case_ids", action="append", help="Explicit subset/smoke case ID"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.k <= 50:
        parser.error("Use --k in 1..50")
    report = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "stage": "validation",
        "completed_cases": 0,
        "aggregates": None,
    }
    owned = False

    def save():
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )

    def stage(value):
        report["stage"] = value
        save()
        print(f"Stage: {value}", flush=True)

    def progress(done, total, id_):
        report["completed_cases"] = done
        save()
        print(f"Completed {done}/{total}: {id_}", flush=True)

    try:
        dataset = load_dataset(args.dataset)
        selected = select_cases(dataset, args.case_ids)
        report.update(
            dataset=dataset.summary(),
            scope="full" if args.case_ids is None else "subset",
            question_count=len(selected),
            k=args.k,
        )
        with args.output.open("x", encoding="utf-8") as file:
            file.write(json.dumps(report) + "\n")
        owned = True
        report = run_benchmark(
            dataset, Settings(), k=args.k, case_ids=args.case_ids, progress=progress, stage=stage
        )
        report["table_markdown"] = markdown_table(report)
        save()
        print(report["table_markdown"])
        return 0
    except KeyboardInterrupt:
        report.update(status="interrupted", aggregates=None)
        if owned:
            save()
        return 130
    except Exception as exc:
        logger.error("retrieval_benchmark_failed error=%s", type(exc).__name__)
        if owned:
            report.update(status="failed", error_type=type(exc).__name__, aggregates=None)
            report["table_markdown"] = markdown_table(report)
            save()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
