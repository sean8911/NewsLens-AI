"""Manage an isolated Windows PostgreSQL instance without a system service.

Download/extract official EDB binaries first; see docs/database.md.
Passwords are generated locally and never printed or passed in process arguments.
"""

import argparse
import os
import secrets
import subprocess
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import URL, create_engine, literal, text
from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".local" / "postgres"
DATA = STATE / "data"
PORT = 55432


def run(binary: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    # A background server can inherit pipe handles on Windows. A log file lets
    # pg_ctl exit without communicate() waiting for every server process to close.
    with (STATE / "management.log").open("a", encoding="utf-8") as output:
        return subprocess.run(
            [str(binary), *args],
            check=check,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )


def password(path: Path) -> str:
    if not path.exists():
        path.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "start", "stop", "status"])
    parser.add_argument(
        "--bin-dir", type=Path, default=ROOT / ".local" / "postgresql" / "pgsql" / "bin"
    )
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This helper targets Windows; use your platform's PostgreSQL service manager")
    pg_ctl = args.bin_dir / "pg_ctl.exe"
    if not pg_ctl.is_file():
        parser.error("PostgreSQL binaries not found; extract them or supply --bin-dir")
    if args.action == "setup":
        STATE.mkdir(parents=True, exist_ok=True)
        admin_password = password(STATE / "admin.password")
        if not (DATA / "PG_VERSION").exists():
            run(
                args.bin_dir / "initdb.exe",
                "-D",
                str(DATA),
                "-U",
                "newslens_admin",
                "--encoding=UTF8",
                "--locale=C",
                "--auth=scram-sha-256",
                f"--pwfile={STATE / 'admin.password'}",
            )
            with (DATA / "postgresql.conf").open("a", encoding="utf-8") as config:
                config.write(f"\nlisten_addresses = '127.0.0.1'\nport = {PORT}\ntimezone = 'UTC'\n")
    elif not (DATA / "PG_VERSION").exists():
        parser.error("Local database is not initialized; run setup first")

    status = run(pg_ctl, "-D", str(DATA), "status", check=False)
    if args.action == "status":
        print(
            "Local PostgreSQL is running"
            if status.returncode == 0
            else "Local PostgreSQL is stopped"
        )
        return
    if args.action == "stop":
        if status.returncode == 0:
            run(pg_ctl, "-D", str(DATA), "-m", "fast", "-w", "stop")
        print("Local PostgreSQL stopped; data retained")
        return
    if status.returncode != 0:
        run(pg_ctl, "-D", str(DATA), "-l", str(STATE / "server.log"), "-w", "start")
    if args.action == "setup":
        app_password = password(STATE / "app.password")
        engine = create_engine(
            URL.create(
                "postgresql+pg8000",
                username="newslens_admin",
                password=admin_password,
                host="127.0.0.1",
                port=PORT,
                database="postgres",
            ),
            isolation_level="AUTOCOMMIT",
            hide_parameters=True,
            connect_args={"timeout": 5},
        )
        try:
            with engine.connect() as connection:
                if not connection.execute(
                    text("SELECT 1 FROM pg_roles WHERE rolname = :name"), {"name": "newslens"}
                ).first():
                    quoted_password = literal(app_password).compile(
                        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
                    )
                    connection.exec_driver_sql(
                        f"CREATE ROLE newslens LOGIN PASSWORD {quoted_password}"
                    )
                for name in ("newslens", "newslens_test"):
                    if not connection.execute(
                        text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
                    ).first():
                        # Identifiers come only from the two fixed names above.
                        connection.exec_driver_sql(f'CREATE DATABASE "{name}" OWNER newslens')
        finally:
            engine.dispose()
        env_path = ROOT / ".env"
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        configured = dotenv_values(env_path) if env_path.exists() else {}
        additions = []
        for key, name in (
            ("NEWSLENS_DATABASE_URL", "newslens"),
            ("NEWSLENS_TEST_DATABASE_URL", "newslens_test"),
        ):
            if not configured.get(key):
                additions.append(
                    f"{key}=postgresql+pg8000://newslens:{app_password}@127.0.0.1:{PORT}/{name}"
                )
        if additions:
            env_path.write_text(existing + "\n" + "\n".join(additions) + "\n", encoding="utf-8")
        print(
            "Local databases ready. Missing .env database settings added; "
            "existing values preserved."
        )
    print(f"Local PostgreSQL running at 127.0.0.1:{PORT}; data in .local/postgres/data")


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, SQLAlchemyError, OSError) as exc:
        # Server logs are local; do not echo connection strings or SQL with generated passwords.
        if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 4551:
            raise SystemExit(
                "Windows Application Control blocked a PostgreSQL executable (WinError 4551). "
                "The device administrator must review the blocked binary before it can run. "
                "Database files have been retained."
            ) from None
        raise SystemExit(
            f"Local PostgreSQL operation failed ({type(exc).__name__}). "
            "Check .local/postgres/server.log and the binary path."
        ) from None
