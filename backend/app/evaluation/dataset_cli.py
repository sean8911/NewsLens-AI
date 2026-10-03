import argparse
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from backend.app.evaluation.news_dataset import DEFAULT_DATASET, load_dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate and inspect the Phase 26 news dataset")
    parser.add_argument("command", choices=["validate", "summary", "export-questions"])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--check-database",
        action="store_true",
        help="Explicit read-only comparison with the configured PostgreSQL corpus",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSONL file for export-questions; never overwrite an existing file",
    )
    args = parser.parse_args(argv)
    if args.command == "export-questions" and args.output is None:
        parser.error("export-questions requires --output")
    if args.command != "export-questions" and args.output is not None:
        parser.error("--output is only available for export-questions")
    try:
        dataset = load_dataset(args.dataset)
        report = {"valid": True, **dataset.summary()}
        if args.check_database:
            from backend.app.core.config import Settings
            from backend.app.evaluation.news_dataset_storage import compare_database

            report["database"] = compare_database(dataset, Settings())
            if not report["database"]["matches"]:
                report["valid"] = False
                print(json.dumps(report, ensure_ascii=True))
                return 1
        if args.command == "export-questions":
            data = "".join(
                json.dumps(case.question_input(), ensure_ascii=False) + "\n"
                for case in dataset.questions
            )
            # Exclusive create avoids clobbering the dataset, labels or a user's output.
            with args.output.open("x", encoding="utf-8", newline="\n") as file:
                file.write(data)
            report["exported_questions"] = len(dataset.questions)
            report["contains_answer_labels"] = False
        print(json.dumps(report, ensure_ascii=True))
        return 0
    except KeyboardInterrupt:
        return 130
    except ValidationError as exc:
        print(
            json.dumps(
                {
                    "valid": False,
                    "errors": [
                        {"location": list(item["loc"]), "message": item["msg"]}
                        for item in exc.errors(
                            include_input=False, include_context=False, include_url=False
                        )
                    ],
                },
                ensure_ascii=True,
            )
        )
        return 1
    except Exception as exc:
        # Validation errors can contain full source text; database errors may hold credentials.
        logging.getLogger(__name__).error("dataset_validation_failed error=%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
