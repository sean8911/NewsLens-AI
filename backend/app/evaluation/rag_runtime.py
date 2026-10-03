"""Frozen-corpus, local-model RAG execution, isolated from production storage."""

import time
from datetime import UTC, datetime
from importlib.metadata import version

from qdrant_client import QdrantClient

from backend.app.database.session import create_session_factory
from backend.app.evaluation.cli import evaluation_database
from backend.app.evaluation.news_dataset import fingerprint
from backend.app.evaluation.rag_evaluation import (
    audit_answer,
    automatic_summary,
    evaluation_cases,
)
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import FusionConfig, SearchRequest
from backend.app.schemas.temporal import RecencyConfig
from backend.app.services.article_embeddings import ArticleEmbeddingService
from backend.app.services.chat import answer_question
from backend.app.services.chat_context import collect_evidence
from backend.app.services.hybrid_search import HybridSearchService
from backend.app.services.lexical_search import LexicalSearchService
from backend.app.services.recency import apply_recency
from backend.app.services.reranking import rerank_results
from backend.app.services.semantic_search import SemanticSearchService
from backend.app.services.time_intent import resolve_time_intent
from nlp.embeddings.store import QdrantEmbeddingStore
from nlp.llm.local import language_model_for
from nlp.multilingual.runtime import provider_for, reranker_for


def run_rag_evaluation(dataset, settings, *, case_ids=None, stage=None, progress=None):
    selected, suite = evaluation_cases(dataset, case_ids)
    started = time.perf_counter()
    fusion = FusionConfig(
        candidate_limit=settings.hybrid_candidate_limit,
        rrf_k=settings.hybrid_rrf_k,
        semantic_weight=settings.hybrid_semantic_weight,
        lexical_weight=settings.hybrid_lexical_weight,
    )
    recency = RecencyConfig(
        decay=settings.recency_decay,
        half_life_days=settings.recency_half_life_days,
        strength=settings.recency_strength,
        reranking_weight=settings.recency_reranking_weight,
    )
    threshold = (
        settings.multilingual_chat_min_reranking_score
        if settings.multilingual_enabled
        else settings.chat_min_reranking_score
    )
    pool = max(5, settings.reranking_candidate_limit)
    if stage:
        stage("database_setup")
    with evaluation_database(settings) as engine:
        sessions = create_session_factory(engine)
        with sessions.begin() as session:
            for article in dataset.articles:
                values = article.model_dump()
                values["url"] = str(article.url)
                session.add(
                    ArticleRecord(
                        **values,
                        retrieved_at=dataset.captured_at,
                        cleaning_version="eval-snapshot-v1",
                        processing_status="clean",
                    )
                )
        if stage:
            stage("model_loading_and_indexing")
        provider, reranker = provider_for(settings), reranker_for(settings)
        llm = language_model_for(settings)
        client = QdrantClient(":memory:")
        try:
            store = QdrantEmbeddingStore(client, "phase28_evaluation", provider.dimensions)
            indexing = ArticleEmbeddingService(sessions, provider, store).sync()
            if indexing["errors"] or indexing["indexed"] != len(dataset.articles):
                raise RuntimeError("Frozen corpus indexing incomplete")
            setup_ms = (time.perf_counter() - started) * 1000
            if stage:
                stage("answer_generation")
            rows = []
            categories = {row.case.id: row.category for row in suite.cases}
            for case in selected:
                tick = time.perf_counter()
                # Only label-free inputs cross into the retrieval/generation pipeline.
                inputs = case.question_input()
                context = inputs["time_context"]
                as_of = datetime.fromisoformat(context["as_of"].replace("Z", "+00:00"))
                request = SearchRequest(
                    q=inputs["question"],
                    **inputs["filters"],
                    date_from=context["date_from"],
                    date_to=context["date_to"],
                    recency="auto",
                    method="hybrid",
                    rerank=True,
                    limit=5,
                )
                intent = resolve_time_intent(request, as_of)
                effective = request.model_copy(
                    update={
                        "q": intent.retrieval_query,
                        "date_from": intent.date_from,
                        "date_to": intent.date_to,
                        "recency": "off",
                        "limit": pool,
                    }
                )
                with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
                    with conn.begin():
                        snapshot = create_session_factory(conn)
                        service = HybridSearchService(
                            SemanticSearchService(snapshot, provider, store, as_of=as_of),
                            LexicalSearchService(snapshot, as_of=as_of),
                            fusion,
                        )
                        retrieval = service.search(effective)
                        retrieval = rerank_results(
                            retrieval,
                            effective,
                            snapshot,
                            reranker,
                            pool,
                            as_of=as_of,
                        )
                        retrieval = apply_recency(retrieval, intent, recency, 5)
                        evidence = collect_evidence(retrieval, effective, snapshot, threshold)
                # Release the read transaction before local generation, as production chat does.
                answer = answer_question(
                    inputs["question"],
                    evidence,
                    lambda: llm,
                    reranker.version,
                    as_of=as_of,
                    answer_language="en",
                ).model_copy(update={"time_intent": intent})
                elapsed = (time.perf_counter() - tick) * 1000
                row = {
                    "id": case.id,
                    "question": case.question,
                    "question_type": case.question_type,
                    "adversarial_category": categories.get(case.id),
                    "temporal_review_required": context["interpretation"] != "snapshot"
                    or categories.get(case.id) == "future_information",
                    "reference": case.model_dump(mode="json"),
                    "response": answer.model_dump(mode="json"),
                    "retrieval": retrieval.model_dump(mode="json"),
                    "evidence": [item.prompt_value() for item in evidence],
                    "latency_ms": elapsed,
                    "audit": audit_answer(case, answer, dataset.articles, evidence),
                }
                rows.append(row)
                if progress:
                    progress(rows)
            return {
                "status": "generated_pending_review",
                "protocol_version": "rag-frozen-v1",
                "evaluated_at": datetime.now(UTC).isoformat(),
                "scope": "full" if case_ids is None else "subset",
                "dataset": dataset.summary(),
                "adversarial_sha256": fingerprint(suite.model_dump(mode="json")),
                "case_count": len(rows),
                "cases": rows,
                "automatic_metrics": automatic_summary(rows),
                "quality_metrics": None,
                "setup_ms": setup_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
                "configuration": {
                    "fusion": fusion.model_dump(),
                    "recency": recency.model_dump(),
                    "reranking_candidate_limit": pool,
                    "min_reranking_score": threshold,
                    "answer_language": "en",
                    "chat_max_input_tokens": llm.max_input_tokens,
                    "chat_generation_seconds": llm.max_seconds,
                },
                "provenance": {
                    "storage": "isolated PostgreSQL schema and in-memory Qdrant",
                    "embedding_model_version": provider.version,
                    "reranking_model_version": reranker.version,
                    "llm_version": llm.version,
                    "indexing": indexing,
                    "versions": {
                        name: version(name)
                        for name in (
                            "newslens-ai",
                            "torch",
                            "transformers",
                            "sentence-transformers",
                            "qdrant-client",
                            "SQLAlchemy",
                        )
                    },
                },
                "limitations": [
                    "Development labels are assistant-authored, not held out or adjudicated.",
                    "Automatic checks verify structure, not semantic correctness or entailment.",
                    "Publication timestamps approximate freshness, not ingestion history.",
                    "Latency excludes corpus setup; first answer may include lazy LLM loading.",
                    "English single-turn benchmark; no HTTP, UI, concurrency or conversation test.",
                    "Uses production services with frozen as-of SQL eligibility and snapshot data.",
                    "No semantic quality score is available until an explicit review is supplied.",
                ],
            }
        finally:
            client.close()
