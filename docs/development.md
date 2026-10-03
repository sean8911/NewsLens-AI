# Development and testing

Install development dependencies with `python -m pip install -e '.[dev]'` inside the virtual environment. Frontend dependencies are pinned by `frontend/pnpm-lock.yaml`.

## Python checks

```powershell
.\.venv\Scripts\python.exe -m ruff check backend ingestion nlp tests scripts infra/docker
.\.venv\Scripts\python.exe scripts/run_tests.py --suite offline
```

The offline suite uses controlled providers for neural inference and requires Python dependencies, but no downloaded neural weights or running PostgreSQL server.

Database tests require a dedicated `NEWSLENS_TEST_DATABASE_URL`:

```powershell
.\.venv\Scripts\python.exe scripts/run_tests.py --suite integration
.\.venv\Scripts\python.exe scripts/run_tests.py --suite all
```

Integration fixtures create and remove isolated test schemas. Never point the test setting at production data. The runner rejects empty, skipped, failed or malformed acceptance results and writes diagnostics under `.local/test-results/` by default.

## Frontend checks

Stop any frontend process using the same build directory before building:

```powershell
Push-Location frontend
pnpm lint
pnpm build
pnpm typecheck
pnpm test
Pop-Location
```

The frontend regression suite uses a mock backend and a separate Next.js test server. If the default mock backend port 8010 is occupied, set `$env:NEWSLENS_TEST_BACKEND_PORT = "18010"` before the test command. The test website uses port 3001.

The latest appearance validation passed the production build, ESLint and all 36 frontend tests. These are frontend checks, not a fresh full-backend or live-model acceptance claim. Desktop/mobile browser checks covered saved history, language switching, sources and the composer. Automated pixel comparison against the design reference was not completed.

## CI and evaluation

`.github/workflows/ci.yml` defines workflow linting, Python checks against native PostgreSQL, and frontend lint/build/type/test jobs. A GitHub-hosted run must be observed after upload; local results do not establish a hosted CI pass.

The evaluation fixtures under `backend/app/evaluation/fixtures/` are a fixed test corpus with authored queries. Retrieval and RAG runners are provided through the project CLI entry points. Their presence does not imply a current benchmark or production accuracy result. Tests and evaluation are separate from collecting current news.

Some `scripts/validate_*.py` tools are historical, machine-specific acceptance utilities and may expect earlier local reports or downloaded models. The portable automated entry points are the test runner, frontend scripts and CI workflow above.

## Version control

Commit source, migrations, configuration examples, lockfiles, tests, CI and the maintained documentation. `.gitignore` excludes local credentials, data, model weights, dependency folders, build output and historical development reports. Do not force-add ignored private files.

Optional Compose definitions and Dockerfiles remain in the repository. Docker is not needed for the supported native setup; container runtime acceptance is separate from the native tests.
