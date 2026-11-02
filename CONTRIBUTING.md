# Contributing to NewsLens AI

NewsLens combines collection, retrieval, local model inference and a web interface. Keep changes focused and preserve the distinction between stored evidence, derived indexes and generated answers.

## Set up a development environment

Follow [getting started](docs/getting-started.md) and [model setup](docs/model-setup.md). Install development dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
Push-Location frontend
pnpm install --frozen-lockfile
Pop-Location
```

Use your own private environment configuration. Integration tests need a dedicated PostgreSQL test database, with permission to create and drop isolated test schemas. Keep it separate from application data.

## Report an issue

Include:

- What you expected and what happened.
- The relevant command, page or endpoint and reproducible steps.
- OS and Python/Node versions, plus whether the failure involves local or server-mode Qdrant.
- Sanitized error output. For answer-quality issues, include the corpus date, filters, question and relevant public source excerpts where permitted.

Never include credentials, private connection strings, access tokens, stored conversations containing personal information or unrestricted database exports. Remove sensitive values from logs before sharing them. Avoid putting sensitive security details in a public issue.

## Make a change

1. Create a branch from the current `main`.
2. Inspect the existing service boundaries and tests before adding another implementation.
3. Add migrations for schema changes. Keep PostgreSQL authoritative and make derived-index reconciliation explicit.
4. Preserve citation provenance, evidence dates, language behavior and recoverable errors.
5. Update documentation and add regression coverage for meaningful behavior changes.
6. Explain the problem, resulting behavior and checks in the pull request.

Do not silently change pinned model revisions or mix vectors from different embedding spaces. Ranking or model changes need evaluation evidence, with the corpus and method recorded.

## Validate

```powershell
.\.venv\Scripts\python.exe -m ruff check backend ingestion nlp tests scripts infra/docker
.\.venv\Scripts\python.exe scripts/run_tests.py --suite offline

# Requires NEWSLENS_TEST_DATABASE_URL for a dedicated test database
.\.venv\Scripts\python.exe scripts/run_tests.py --suite all

Push-Location frontend
pnpm lint
pnpm build
pnpm typecheck
pnpm test
Pop-Location
```

Stop the local frontend before building its `.next` directory. If mock backend port 8010 is occupied, set `NEWSLENS_TEST_BACKEND_PORT` to an available port before frontend tests. See [development](docs/development.md) for details.

Report which checks actually ran. Distinguish offline tests, PostgreSQL integration, real model inference, publisher-feed checks and browser checks. Do not report a skipped or blocked check as passed, and do not infer answer correctness from citation validity alone.

## Repository hygiene

Commit code, migrations, tests, lockfiles, safe configuration examples and maintained documentation. Keep `.env`, model weights, local news exports, database state, caches, handovers and generated reports out of source control. Follow `.gitignore`; do not force-add private files.

Preserve upstream model and asset licenses. Keep article source links and attribution, and respect publisher extraction policies. Do not add fabricated results, sample news presented as current data, or unmeasured performance claims.
