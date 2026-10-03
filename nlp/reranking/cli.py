import argparse
import json
import logging

from backend.app.core.config import Settings
from nlp.reranking.model import CrossEncoderReranker


def reranker_for(settings: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        settings.reranking_model_path,
        settings.reranking_model_id,
        settings.reranking_model_revision,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the optional local cross-encoder")
    parser.add_argument("command", choices=["download"])
    parser.parse_args(argv)
    try:
        provider = reranker_for(Settings())
        provider.download()
        print(
            json.dumps(
                {
                    "status": "ready",
                    "model": provider.model_id,
                    "revision": provider.revision,
                    "model_version": provider.version,
                }
            )
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error("reranker_download_failed error=%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
