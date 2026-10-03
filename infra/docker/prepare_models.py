"""Explicit model setup; separate processes release memory between downloads."""

import subprocess
import sys


def main():
    for module in (
        "nlp.classification.cli",
        "nlp.entities.cli",
        "nlp.embeddings.cli",
        "nlp.reranking.cli",
        "nlp.multilingual.cli",
        "backend.app.chat_cli",
    ):
        subprocess.run([sys.executable, "-m", module, "download"], check=True)


if __name__ == "__main__":
    main()
