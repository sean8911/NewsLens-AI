import argparse
import json
import logging

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.event_runtime import (
    clustering_service,
    event_reader,
    sync_event_summaries,
    sync_events,
)
from backend.app.schemas.event import EventListRequest
from backend.app.schemas.event_timeline import TimelineRequest


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer")
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Cluster reports of the same event and inspect decisions"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sync", help="Reconcile all eligible indexed articles atomically")
    summaries = commands.add_parser("summarize", help="Refresh grounded summaries from SQL sources")
    summaries.add_argument("--event-id", type=positive)
    summaries.add_argument("--refresh", action="store_true")
    listing = commands.add_parser("list")
    listing.add_argument("--limit", type=positive, default=20)
    listing.add_argument("--offset", type=int, default=0)
    listing.add_argument("--min-articles", type=positive, default=2)
    detail = commands.add_parser("show")
    detail.add_argument("event_id", type=positive)
    compare = commands.add_parser("compare")
    compare.add_argument("article_a", type=positive)
    compare.add_argument("article_b", type=positive)
    source_comparison = commands.add_parser(
        "compare-sources", help="Compare cited reporting across publishers for one event"
    )
    source_comparison.add_argument("event_id", type=positive)
    timeline = commands.add_parser("timeline", help="Read cited reports in publication order")
    timeline.add_argument("event_id", type=positive)
    timeline.add_argument("--limit", type=positive, default=50)
    timeline.add_argument("--offset", type=int, default=0)
    history = commands.add_parser("runs")
    history.add_argument("--limit", type=positive, default=5)
    args = parser.parse_args(argv)
    try:
        settings = Settings()
        configure_logging(settings.log_level)
        if args.command == "sync":
            result = sync_events(settings)
        elif args.command == "summarize":
            result = sync_event_summaries(settings, event_id=args.event_id, force=args.refresh)
        elif args.command == "compare":
            with clustering_service(settings) as service:
                result = service.compare_articles(args.article_a, args.article_b)
        else:
            with event_reader(settings) as service:
                if args.command == "list":
                    result = service.list(
                        EventListRequest(
                            limit=args.limit, offset=args.offset, min_articles=args.min_articles
                        )
                    ).model_dump(mode="json")
                elif args.command in {"show", "compare-sources", "timeline"}:
                    event = (
                        service.timeline(
                            args.event_id, TimelineRequest(limit=args.limit, offset=args.offset)
                        )
                        if args.command == "timeline"
                        else service.comparison(args.event_id)
                        if args.command == "compare-sources"
                        else service.get(args.event_id)
                    )
                    if event is None:
                        print(json.dumps({"error": "Event not found"}))
                        return 1
                    result = event.model_dump(mode="json")
                else:
                    result = service.runs(args.limit)
        print(json.dumps(result, ensure_ascii=True))
        return (
            1
            if isinstance(result, dict)
            and (
                result.get("errors")
                or result.get("status") in {"error", "not_found"}
                or result.get("summaries", {}).get("errors")
            )
            else 0
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error(
            "event_command_failed error=%s; check migrations and index", type(exc).__name__
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
