import argparse
import logging

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.schemas.trending import TrendingRequest
from backend.app.trending_runtime import read_trending


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank stored topics by coverage and growth")
    parser.add_argument("--as-of", help="Timezone-aware ISO timestamp; defaults to now")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--group-by", choices=["topic", "entity_topic"], default="topic")
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        request = TrendingRequest(**vars(args))
        settings = Settings()
        configure_logging(settings.log_level)
        result = read_trending(request, settings)
        print(result.model_dump_json())
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error("trending_failed error=%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
