"""Summarize content-free JSONL telemetry, including mixed stderr logs."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from backend.app.core.performance import STAGES

TIMINGS = tuple(f"{stage}_time_ms" for stage in STAGES) + ("total_time_ms",)
USAGE = (
    "llm_input_tokens",
    "llm_output_tokens",
    "embedding_input_tokens",
    "llm_calls",
    "embedding_calls",
    "llm_completed_calls",
    "embedding_completed_calls",
    "llm_api_cost_usd",
    "embedding_api_cost_usd",
)


def summarize(lines) -> dict:
    groups = defaultdict(list)
    ignored = 0
    ids = set()
    for line in lines:
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            ignored += 1
            continue
        if not isinstance(row, dict) or row.get("event") != "performance":
            ignored += 1
            continue
        try:
            if row["schema_version"] != 1 or row["operation_id"] in ids:
                raise ValueError("Unsupported version or duplicate measurement")
            if not isinstance(row["operation"], str) or not isinstance(row["outcome"], str):
                raise ValueError("Invalid measurement labels")
            for value in [
                *(row[key] for key in TIMINGS),
                *(row["usage"][key] for key in USAGE),
                *(row["stage_failures"][key] for key in STAGES),
            ]:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("Invalid measurement number")
            ids.add(row["operation_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid performance record") from exc
        groups[row["operation"]].append(row)
    if not groups:
        raise ValueError("No performance records found")
    result = {}
    for name, rows in sorted(groups.items()):
        timings = {}
        for key in TIMINGS:
            values = sorted(row[key] for row in rows)
            timings[key] = {
                "mean": round(mean(values), 3),
                "p50": round(median(values), 3),
                "p95": values[math.ceil(len(values) * 0.95) - 1],
                "max": values[-1],
            }
        result[name] = {
            "count": len(rows),
            "outcomes": dict(Counter(row["outcome"] for row in rows)),
            "timings_ms": timings,
            "stage_failures": {
                key: sum(row["stage_failures"][key] for row in rows) for key in STAGES
            },
            "usage_totals": {key: sum(row["usage"][key] for row in rows) for key in USAGE},
        }
    return {
        "schema_version": 1,
        "record_count": len(ids),
        "ignored_lines": ignored,
        "percentile_method": "p50 median; p95 nearest rank; zeros include unexecuted stages",
        "cost_basis": "local_inference_no_external_api_charges; hardware cost unknown",
        "operations": result,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="JSONL telemetry or mixed UTF-8 stderr log")
    parser.add_argument("--output", type=Path, help="Create a new JSON report; never overwrite")
    args = parser.parse_args(argv)
    try:
        with args.log.open(encoding="utf-8-sig") as stream:
            result = summarize(stream)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        else:
            print(payload, end="")
        return 0
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"Performance report failed: {type(exc).__name__}.\n")


if __name__ == "__main__":
    raise SystemExit(main())
