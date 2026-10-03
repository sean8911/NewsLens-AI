# Getting started

Run commands from the repository root unless a step says otherwise. Examples use PowerShell on Windows. Docker is not required.

## Install dependencies

Install Python 3.11+, PostgreSQL, and Node.js satisfying `frontend/package.json` (Node.js 24 is supported by the project). Install the pinned package manager if needed:

```powershell
npm install -g pnpm@11.19.0
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
```

Create configuration only if it does not already exist:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
Push-Location frontend
pnpm install --frozen-lockfile
Pop-Location
```

The root `.env` is private and must not be committed. Environment variables override its values. The supplied examples contain placeholders, not working database credentials.

## Prepare data and models

1. Complete [database setup](database.md), including `alembic upgrade head`.
2. Follow [model preparation](model-setup.md) to download the configured snapshots.
3. Collect the first batch and synchronize indexes using the commands in that guide.

An empty database can display the interface, but it has no evidence for answers. Downloading models alone does not collect news.

## Start the application

Keep the following three terminals open. Virtual-environment activation is optional because these commands use its Python directly.

### Terminal 1: database and automatic updates

For the isolated Windows database provisioned by the helper:

```powershell
.\.venv\Scripts\python.exe scripts/local_postgres.py start
$env:NEWSLENS_SCHEDULER_ENABLED = "true"
$env:NEWSLENS_SCHEDULER_RUN_IMMEDIATELY = "true"
.\.venv\Scripts\python.exe -m ingestion.scheduler run
```

If using a separately installed PostgreSQL service, start that service instead of the helper. The updater runs immediately, then at the interval configured by `NEWSLENS_SCHEDULER_INTERVAL_SECONDS` (600 seconds in the example configuration).

### Terminal 2: backend

```powershell
.\.venv\Scripts\python.exe -m backend.app.api_cli --port 18000
```

API documentation: <http://127.0.0.1:18000/docs>.

### Terminal 3: website

```powershell
$env:NEWSLENS_API_URL = "http://127.0.0.1:18000"
.\.venv\Scripts\python.exe scripts/run_frontend.py
```

Open <http://127.0.0.1:3000/chat>. The launcher uses installed Node.js and frontend dependencies; a globally available pnpm command is not needed for subsequent starts. It refuses an occupied website port to prevent accidental duplicate servers.

The frontend defaults to backend port 8000 when `NEWSLENS_API_URL` is unset. Set the variable as shown every time you start on 18000, or put it in an untracked `frontend/.env.local`. This server-only value is not a browser API key.

Stop a terminal process with Ctrl+C. Closing the updater stops automatic collection. Stop the isolated database separately with `scripts/local_postgres.py stop` when it is no longer in use.

## Keeping news current

```powershell
.\.venv\Scripts\python.exe -m ingestion.scheduler status
.\.venv\Scripts\python.exe -m ingestion.scheduler once
```

`status` inspects persisted collection outcomes; `once` performs one update. Feed configuration is in `ingestion/feeds/defaults.json`. Successful collection is distinct from successful vector indexing. Inspect update failures and prepare unavailable models before relying on semantic retrieval.

The website shows collection freshness and each answer's evidence dates. After a new collection, ask a new question or follow-up to retrieve evidence again. Saved answers remain historical records.

## Common startup problems

| Symptom | Action |
| --- | --- |
| Port already in use | Open the existing instance or choose a free port. Do not start the backend twice on the same port. |
| `pnpm` not recognized | Install it for dependency setup; use the Python frontend launcher for later starts. |
| Database connection failure | Check the configured PostgreSQL service and private `.env` database URL. |
| Missing model or unavailable inference | Complete model downloads and verify the PyTorch runtime. |
| Windows blocks an executable or DLL | Use an administrator-approved runtime; do not treat a frontend-only check as a working model check. |
| Insufficient information | Inspect corpus coverage, selected filters and cited dates; collect/index relevant sources. |

For a production frontend build, stop the website process, run `pnpm build` inside `frontend/`, then use the same API environment setting with `scripts/run_frontend.py --production`. This serves the local production build; it does not deploy the site.
