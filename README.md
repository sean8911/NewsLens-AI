# NewsLens AI

A source-grounded news intelligence platform for collecting reporting, finding relevant articles, and asking questions with inspectable citations.

NewsLens combines a FastAPI backend, PostgreSQL, Qdrant retrieval, local NLP models and a Next.js interface. The chat and shared navigation support English and Chinese. Search and answer processing support English, Malay and Chinese.

## Features

- **News collection:** configurable RSS feeds, duplicate detection, content cleaning and scheduled updates with persisted collection status.
- **Search:** PostgreSQL keyword search and vector search, combined with reciprocal rank fusion, optional cross-encoder reranking and recency-aware ranking.
- **Grounded chat:** evidence selection, numbered citations, source excerpts, saved conversations and contextual follow-up questions.
- **Article analysis:** language detection, topics, named entities, keywords and extractive summaries.
- **Event intelligence:** related-article clustering, cited summaries, publisher comparisons, timelines and trending topics.
- **Responsive interface:** Metro Focus styling, English/Chinese controls, mobile layouts and visible publication and collection dates.

## How it works

```text
RSS feeds -> cleaning and article processing -> PostgreSQL
                           |                       |
                      embeddings              keyword search
                           |                       |
                         Qdrant ------ hybrid retrieval
                                             |
                                    reranking + evidence
                                             |
                                    local model selection
                                             |
                                   validated cited answer
                                             |
                                    FastAPI -> Next.js
```

The chat model selects evidence from retrieved articles. The server validates those selections and assembles excerpts and citations; translated excerpts retain their original evidence. NewsLens can return insufficient information when the collected sources do not support an answer.

## Technology

| Layer | Implementation |
| --- | --- |
| Web application | Next.js, React, TypeScript, Tailwind CSS, Lucide |
| API | FastAPI, Pydantic, Uvicorn |
| Data | PostgreSQL, SQLAlchemy, Alembic |
| Retrieval | Qdrant, Sentence Transformers, PostgreSQL full-text search |
| Models | PyTorch, ONNX Runtime, multilingual embeddings, reranking, translation and a local Qwen model |
| Collection | feedparser, Requests, APScheduler |
| Validation | pytest, Ruff, Playwright, ESLint, TypeScript, GitHub Actions |

## Run locally

Docker is optional. The native setup uses Python 3.11+, PostgreSQL, Node.js compatible with `frontend/package.json`, and pnpm matching its `packageManager` field.

1. Follow [installation and startup](docs/getting-started.md).
2. Configure [PostgreSQL](docs/database.md) and prepare the [local models](docs/model-setup.md).
3. Start the updater, backend and website in separate terminals.
4. Open **http://127.0.0.1:3000/chat**.

The startup guide uses API port **18000** and website port **3000**. If either is occupied, stop your existing instance or choose another port and keep the frontend API setting consistent.

## Repository structure

```text
backend/       API, retrieval, chat, event services and database migrations
frontend/      Next.js application, assets and frontend tests
ingestion/     Feed collection, cleaning, extraction and update scheduling
nlp/           Language, article analysis, embeddings and model adapters
tests/         Python unit and PostgreSQL integration tests
scripts/       Local startup, validation and development utilities
infra/         Optional container configuration
requirements/  CI dependency constraints
docs/          Setup, architecture and development guides
.github/       Continuous integration workflow
```

## Documentation

- [Getting started](docs/getting-started.md)
- [Database setup](docs/database.md)
- [Local models and indexing](docs/model-setup.md)
- [Architecture and boundaries](docs/architecture.md)
- [Development and testing](docs/development.md)

## Scope and limitations

NewsLens answers from its collected corpus. It does not browse the live web for each chat question and has no dedicated live weather or product-price service. Keeping the updater running refreshes the corpus; it does not rewrite previous answers. Coverage depends on configured publishers and available article text.

Local model downloads and a working PyTorch runtime are required for neural search and chat. A successful frontend build does not establish model availability. Answer quality depends on retrieval, source coverage and model selection, so citations should be inspected. Historical evaluation data is a fixed test corpus, not a current-news feed or a claim of production accuracy.

The application is designed for local use. Public hosting requires deployment-specific authentication, access controls, TLS and operational review. Collected articles remain attributable to their publishers; model and bundled asset licenses remain applicable.
