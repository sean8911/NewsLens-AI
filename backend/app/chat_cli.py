import argparse
import json
import logging
from uuid import UUID

from backend.app.chat_runtime import run_chat
from backend.app.conversation_runtime import create_conversation, read_conversation
from backend.app.core.config import Settings
from backend.app.schemas.chat import ChatRequest
from nlp.llm.local import language_model_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions grounded in collected news")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Explicitly prepare the pinned local LLM")
    commands.add_parser("conversation-new", help="Create a persistent conversation")
    history = commands.add_parser("history", help="Read a page of paired conversation turns")
    history.add_argument("conversation_id", type=UUID)
    history.add_argument("--limit", type=int, default=20, choices=range(1, 51))
    history.add_argument("--before", type=int)
    ask = commands.add_parser("ask")
    ask.add_argument("--recency", choices=["auto", "off", "latest"], default=argparse.SUPPRESS)
    ask.add_argument("--conversation-id", type=UUID)
    ask.add_argument("--expected-revision", type=int)
    ask.add_argument("--follow-up", action=argparse.BooleanOptionalAction, default=None)
    ask.add_argument("--reset-context", action="store_true")
    ask.add_argument("question")
    ask.add_argument("--answer-language", choices=["en", "ms", "zh"])
    for field in ("source", "language", "category", "date-from", "date-to"):
        ask.add_argument("--" + field, default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    request = None
    if args.command == "ask":
        try:
            request = ChatRequest.model_validate(
                {k: v for k, v in vars(args).items() if k != "command"}
            )
        except ValueError:
            parser.error("Invalid question or filters; see --help")
    try:
        if args.command == "download":
            provider = language_model_for(Settings())
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
        elif args.command == "conversation-new":
            print(create_conversation().model_dump_json())
        elif args.command == "history":
            print(
                read_conversation(args.conversation_id, args.limit, args.before).model_dump_json()
            )
        else:
            print(run_chat(request).model_dump_json(ensure_ascii=True))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error("chat_failed error=%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
