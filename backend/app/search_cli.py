import argparse
import logging

from backend.app.core.logging import configure_logging
from backend.app.schemas.search import SearchRequest
from backend.app.search_runtime import run_search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search news by meaning or keywords")
    parser.add_argument("q", help="Natural-language or keyword query")
    parser.add_argument("--method", choices=["semantic", "lexical", "hybrid"], default="semantic")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recency", choices=["auto", "off", "latest"], default="auto")
    parser.add_argument("--source")
    parser.add_argument("--language")
    parser.add_argument("--category")
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    args = parser.parse_args(argv)
    configure_logging("INFO")
    try:
        request = SearchRequest.model_validate(vars(args))
    except ValueError:
        parser.error("Invalid query or filters; use --help and the semantic search guide")
    try:
        print(run_search(request).model_dump_json())
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error(
            "%s_search_failed error=%s", request.method, type(exc).__name__
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
