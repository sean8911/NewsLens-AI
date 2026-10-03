"""Start the NewsLens website without requiring pnpm on the terminal PATH."""

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument(
        "--production", action="store_true", help="Use an existing production build"
    )
    parser.add_argument("--check", action="store_true", help="Check startup requirements and exit")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1..65535")
    frontend = ROOT / "frontend"
    next_cli = frontend / "node_modules" / "next" / "dist" / "bin" / "next"
    bundled = (
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
    )
    node = shutil.which("node") or (str(bundled) if bundled.is_file() else None)
    if not node:
        print(
            "Node.js was not found. Install Node.js 24 and reopen your terminal.", file=sys.stderr
        )
        return 1
    if not next_cli.is_file():
        print(
            "Frontend dependencies are missing. Install them with pnpm install in frontend.",
            file=sys.stderr,
        )
        return 1
    if args.production and not (frontend / ".next/BUILD_ID").is_file():
        print(
            "No production build found. Omit --production to use development mode.", file=sys.stderr
        )
        return 1
    # Explicitly refuse duplicate startup rather than silently choosing a different port.
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
    except OSError:
        print(
            f"Port {args.port} is already in use. If NewsLens is running, open "
            f"http://127.0.0.1:{args.port}/chat instead of starting it twice.",
            file=sys.stderr,
        )
        return 1
    print(f"NewsLens chat: http://127.0.0.1:{args.port}/chat", flush=True)
    if args.check:
        print("Node.js and frontend dependencies found; port is available.")
        return 0
    command = [node, str(next_cli), "start" if args.production else "dev"]
    if not args.production:
        command.append("--webpack")
    command.extend(["--hostname", "127.0.0.1", "--port", str(args.port)])
    env = os.environ.copy()
    env["PATH"] = str(Path(node).parent) + os.pathsep + env.get("PATH", "")
    try:
        return subprocess.call(command, cwd=frontend, env=env)
    except KeyboardInterrupt:
        return 130
    except OSError:
        print("Frontend startup failed. Check the Node.js installation.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
