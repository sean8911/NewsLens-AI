# Database setup

PostgreSQL is the source of truth for articles, processing records, collection outcomes, conversations and event data. Qdrant stores derived vectors separately. Schema changes are managed by Alembic; API startup does not create tables automatically.

## Existing PostgreSQL service

Create an application database and role with permission to own its tables. Configure your private root `.env`:

```dotenv
NEWSLENS_DATABASE_URL=postgresql+pg8000://YOUR_USER:YOUR_PASSWORD@127.0.0.1:5432/newslens
NEWSLENS_TEST_DATABASE_URL=postgresql+pg8000://YOUR_USER:YOUR_PASSWORD@127.0.0.1:5432/newslens_test
```

Replace placeholders and URL-encode special characters in passwords. The test database must be separate from application data; its role needs permission to create and drop test schemas. Remote connections support certificate verification through `NEWSLENS_DATABASE_SSL` and the optional CA-file setting in `.env.example`.

Apply migrations from the repository root:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
```

## Isolated Windows installation

The optional helper manages an isolated PostgreSQL instance on loopback port **55432**, without a system service. It does not download PostgreSQL.

1. Obtain official Windows PostgreSQL binaries and extract them so `.local/postgresql/pgsql/bin/pg_ctl.exe` exists. Alternatively pass the binary directory through `--bin-dir`.
2. Install the Python dependencies and create `.env` as described in [getting started](getting-started.md).
3. Run:

```powershell
.\.venv\Scripts\python.exe scripts/local_postgres.py setup
.\.venv\Scripts\python.exe -m alembic upgrade head
```

The helper generates credentials locally, creates `newslens` and `newslens_test`, and adds missing database settings to `.env`. It preserves already configured values: check that an existing setting points to the server you intend to use. Credentials and database files stay under ignored local paths.

```powershell
.\.venv\Scripts\python.exe scripts/local_postgres.py status
.\.venv\Scripts\python.exe scripts/local_postgres.py start
.\.venv\Scripts\python.exe scripts/local_postgres.py stop
```

Stopping retains database files. Back up the database before schema or infrastructure changes. Never upload `.env`, `.local/`, database dumps, passwords or stored conversations to the repository.

When upgrading data collected by an earlier cleaning implementation, run the explicit backfill if the updater reports pending cleaning:

```powershell
.\.venv\Scripts\python.exe -m ingestion.cleaners.cli backfill
```
