# Local models and indexing

Model weights are downloaded explicitly and stored under `.local/models/`; they are not part of the repository. Model IDs, pinned revisions, paths and limits are defined in `.env.example` and the runtime configuration. Downloads require network access, disk space and a compatible local runtime.

## Prepare configured models

From the repository root after installing Python dependencies:

```powershell
.\.venv\Scripts\python.exe -m nlp.classification.cli download
.\.venv\Scripts\python.exe -m nlp.entities.cli download
.\.venv\Scripts\python.exe -m nlp.embeddings.cli download
.\.venv\Scripts\python.exe -m nlp.reranking.cli download
.\.venv\Scripts\python.exe -m nlp.multilingual.cli download
.\.venv\Scripts\python.exe -m backend.app.chat_cli download
```

| Component | Purpose |
| --- | --- |
| Topic and entity models | Article topics and named entities |
| all-MiniLM-L6-v2 | Legacy article vectors used for event clustering |
| ms-marco-MiniLM cross-encoder | Reranking for the non-multilingual retrieval path |
| multilingual-e5-small | Shared English/Malay/Chinese retrieval space |
| bge-reranker-v2-m3 | Multilingual relevance reranking |
| M2M100 418M | Translation of selected evidence |
| Qwen2.5-1.5B-Instruct | Local evidence selection and bounded chat/research operations |

The multilingual download command prepares its embedding, reranking and translation models. The other commands prepare their own snapshots. Model licenses and operating requirements remain those of the model publishers. No commercial hosted LLM key is required by these local providers.

## Collect and index

After database migrations and model preparation:

```powershell
.\.venv\Scripts\python.exe -m ingestion.scheduler once
.\.venv\Scripts\python.exe -m nlp.embeddings.cli backfill
.\.venv\Scripts\python.exe -m nlp.multilingual.cli backfill
```

Ingestion also attempts derived processing and indexing after committing articles. Explicit backfills reconcile existing rows or recover from unavailable models. A successful article commit is not proof that its vectors are ready.

Qdrant uses `.local/qdrant` by default, so no separate Qdrant server or Docker container is required. Set `NEWSLENS_QDRANT_URL` for server mode. Use distinct collection prefixes when separate databases share a Qdrant instance. Local index access is coordinated between the API, updater and CLI; a long update can produce a retryable busy response.

## Check an answer

Ask a question about a title or fact that is actually present in your collected corpus:

```powershell
.\.venv\Scripts\python.exe -m backend.app.chat_cli ask "What does the collected reporting say about artificial intelligence?"
```

The command can legitimately return insufficient information. An answered response should have supporting source IDs and excerpts; inspect them rather than treating response generation alone as a quality test.

The interface language selector changes chat controls, shared labels and the requested language for new answers. It does not translate stored articles or rewrite saved conversations. API answer-language selection supports `en`, `ms` and `zh`.

Model imports or inference can fail independently of the website. Verify PyTorch and model loading on the target machine before claiming end-to-end availability. The repository does not include model files or a prepopulated production database.
