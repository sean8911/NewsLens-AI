import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.evaluation.news_dataset import DEFAULT_DATASET, load_dataset
from backend.app.evaluation.rag_evaluation import evaluation_cases, review_template, score_review
from backend.app.evaluation.rag_runtime import run_rag_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate frozen-corpus RAG and review real answers"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--case", dest="case_ids", action="append")
    template = commands.add_parser("review-template")
    template.add_argument("--report", type=Path, required=True)
    score = commands.add_parser("score-review")
    score.add_argument("--report", type=Path, required=True)
    score.add_argument("--review", type=Path, required=True)
    for command in (run, template, score):
        command.add_argument("--output", type=Path, required=True, help="New file; never overwrite")
    args = parser.parse_args(argv)
    report = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "stage": "validation",
        "completed_cases": 0,
        "cases": [],
        "automatic_metrics": None,
        "quality_metrics": None,
    }
    owned = False

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def stage(value):
        report["stage"] = value
        save()
        print(f"Stage: {value}", flush=True)

    def progress(rows):
        report.update(completed_cases=len(rows), cases=list(rows))
        save()
        print(f"Completed {len(rows)}/{report['case_count']}: {rows[-1]['id']}", flush=True)

    try:
        if args.command == "run":
            dataset = load_dataset(args.dataset)
            cases, _ = evaluation_cases(dataset, args.case_ids)
            report.update(
                dataset=dataset.summary(),
                case_count=len(cases),
                scope="full" if args.case_ids is None else "subset",
            )
        else:
            source = json.loads(args.report.read_text(encoding="utf-8"))
            report = (
                review_template(source)
                if args.command == "review-template"
                else score_review(source, json.loads(args.review.read_text(encoding="utf-8")))
            )
        with args.output.open("x", encoding="utf-8") as file:
            file.write(json.dumps(report, indent=2) + "\n")
        owned = True
        if args.command == "run":
            report = run_rag_evaluation(
                dataset,
                Settings(),
                case_ids=args.case_ids,
                stage=stage,
                progress=progress,
            )
            save()
        print(f"Saved {args.command} output; status={report.get('status', 'review_template')}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        logging.getLogger(__name__).error("rag_evaluation_failed error=%s", type(exc).__name__)
        if owned:
            report.update(
                status="interrupted" if interrupted else "failed",
                error_type=type(exc).__name__,
                automatic_metrics=None,
                quality_metrics=None,
            )
            save()
        return 130 if interrupted else 1


if __name__ == "__main__":
    raise SystemExit(main())
