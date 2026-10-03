import argparse
import logging

from backend.app.research_runtime import run_research
from backend.app.schemas.research import ResearchRequest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Investigate collected news with cited evidence")
    parser.add_argument("question")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-source", dest="compare_sources", action="append", default=[])
    parser.add_argument("--answer-language", choices=["en", "ms", "zh"])
    parser.add_argument("--recency", choices=["auto", "off", "latest"], default="auto")
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--max-articles", type=int, default=8)
    for field in ("source", "language", "category", "date-from", "date-to"):
        parser.add_argument("--" + field)
    args = parser.parse_args(argv)
    try:
        request = ResearchRequest.model_validate(vars(args))
    except ValueError:
        parser.error("Invalid question, filters or research bounds; see --help")
    try:
        print(run_research(request).model_dump_json(ensure_ascii=True))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error("research_failed error=%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
