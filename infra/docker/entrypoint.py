"""Construct container-only connection settings without logging credentials."""

import os
import sys

from sqlalchemy.engine import URL


def database_url(password: str) -> str:
    return URL.create(
        "postgresql+pg8000",
        username="newslens",
        password=password,
        host="postgres",
        port=5432,
        database="newslens",
    ).render_as_string(hide_password=False)


def main():
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise SystemExit("POSTGRES_PASSWORD must be set for the container database")
    if len(sys.argv) < 2:
        raise SystemExit("A container command is required")
    os.environ["NEWSLENS_DATABASE_URL"] = database_url(password)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
