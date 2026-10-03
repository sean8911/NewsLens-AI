# Architecture

## Collection and storage

The collector reads the configured RSS/Atom feeds, validates metadata and normalizes article URLs. Cleaning, content extraction and processing records preserve provenance. PostgreSQL retains articles, update outcomes, NLP results, vector synchronization records, conversations and event relationships.

APScheduler runs a separate update worker. Starting the API or refreshing the website's status panel does not start collection. Feed content is used first; optional webpage extraction is controlled by source policies and configuration.

## Retrieval-augmented chat

1. Validate the question, language, source/date filters and conversation revision.
2. Retrieve candidates using PostgreSQL full-text search and Qdrant semantic search.
3. Combine ranks using reciprocal rank fusion, apply requested recency behavior, and rerank candidates with a cross-encoder.
4. Build bounded evidence from article titles, descriptions and available bodies with source identifiers and timestamps.
5. Ask the local model to select supporting evidence. Validate the returned identifiers against the supplied context.
6. Construct cited excerpts, translate selected text when requested, and return inspectable source metadata. Abstain or report an error when evidence or model output is unusable.

The model is not a general live-web assistant. The answer contract emphasizes selected evidence, including translations with original excerpts, rather than unrestricted prose generation. A reranker score is a relevance heuristic, not a calibrated correctness probability.

Follow-ups use bounded prior context and retrieve evidence again. Saved answers keep their original evidence dates. A new collection does not alter old conversation records.

## Article and event intelligence

Language detection, topic classification, entity extraction, keywords and extractive summaries enrich articles. Embeddings support related-article clustering. Event services expose cited summaries, publisher comparison, chronological reporting and related entities. Trending scores describe activity in the collected corpus, not the whole internet.

## Web and API boundary

The Next.js frontend accesses FastAPI through its same-origin server proxy. `NEWSLENS_API_URL` is server-only. Request validation, bounded payloads, rate limits, inference admission and sanitized errors support the local workflow.

The Metro Focus interface uses a black header, white reading area and blue history rail. On mobile, recent conversations scroll horizontally and workspace notes move below the conversation. The English/Chinese interface preserves original source text.

## Operational boundaries

- PostgreSQL is authoritative; vector indexes and model-derived results can require reconciliation.
- Collection timestamps and publication timestamps answer different questions and remain separate.
- Source coverage, extraction quality, retrieval and local model behavior limit answers.
- Public hosting needs authentication, TLS, appropriate network restrictions and deployment-specific operations.
- Model files, secrets, local news exports, conversation data, caches and generated diagnostics are excluded from version control.
