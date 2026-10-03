import argparse
import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from qdrant_client import QdrantClient
from sqlalchemy import text

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.evaluation.dataset import EvaluationDataset
from backend.app.evaluation.metrics import retrieval_metrics
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import FusionConfig, SearchRequest
from backend.app.services.article_embeddings import ArticleEmbeddingService
from backend.app.services.hybrid_search import fuse_results
from backend.app.services.lexical_search import LexicalSearchService
from backend.app.services.reranking import rerank_results
from backend.app.services.semantic_search import SemanticSearchService
from nlp.embeddings.service import digest
from nlp.embeddings.store import QdrantEmbeddingStore
from nlp.multilingual.runtime import provider_for, reranker_for

DEFAULT_DATASET = Path(__file__).parent / "fixtures" / "hybrid.json"
ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def evaluation_database(settings: Settings):
    if settings.test_database_url is None:
        raise ValueError("Set NEWSLENS_TEST_DATABASE_URL for isolated evaluation")
    schema = "newslens_eval_" + uuid4().hex
    admin_settings = Settings(_env_file=None, database_url=settings.test_database_url)
    admin = create_database_engine(admin_settings)
    engine = None
    created = False
    try:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            created = True
        db_settings = Settings(
            _env_file=None, database_url=settings.test_database_url, database_schema=schema
        )
        engine = create_database_engine(db_settings)
        config = Config(str(ROOT / "alembic.ini"))
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        if created:
            # Only the unique schema created by this invocation may be removed.
            with admin.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def evaluate(
    dataset: dict,
    settings: Settings,
    configs: dict[str, FusionConfig],
    methods: list[str],
    k: int,
    compare_reranking: bool = False,
) -> dict:
    EvaluationDataset.model_validate(dataset)
    if not methods or set(methods) - {"semantic", "lexical", "hybrid"} or not configs:
        raise ValueError("Select supported retrieval methods and at least one configuration")
    if not 1 <= k <= 50 or set(configs) & {"semantic", "lexical"}:
        raise ValueError("Invalid cutoff or reserved configuration name")
    if any(name.endswith("_reranked") for name in configs):
        raise ValueError("Reserved reranking configuration suffix")
    reranker = reranker_for(settings) if compare_reranking else None
    needs_vectors = bool(set(methods) & {"semantic", "hybrid"})
    provider = provider_for(settings) if needs_vectors else None
    client = QdrantClient(":memory:") if needs_vectors else None
    cases = []
    try:
        store = QdrantEmbeddingStore(client, "evaluation", provider.dimensions) if client else None
        with evaluation_database(settings) as engine:
            sessions = create_session_factory(engine)
            with sessions.begin() as session:
                for document in dataset["documents"]:
                    session.add(
                        ArticleRecord(
                            **document,
                            url=f"https://example.com/evaluation/{document['id']}",
                            source="Synthetic evaluation",
                            category="Technology",
                            language="en",
                            retrieved_at=datetime.now(UTC),
                            cleaning_version="1",
                            processing_status="clean",
                        )
                    )
            if needs_vectors:
                totals = ArticleEmbeddingService(sessions, provider, store).sync()
                if totals["errors"]:
                    raise RuntimeError("Evaluation indexing failed")
            semantic = SemanticSearchService(sessions, provider, store) if needs_vectors else None
            lexical = LexicalSearchService(sessions)
            depth = max(config.candidate_limit for config in configs.values())
            depth = max(depth, k, settings.reranking_candidate_limit if compare_reranking else 1)
            for case in dataset["queries"]:
                request = SearchRequest(q=case["q"], limit=depth)
                sem = semantic.search(request) if set(methods) & {"semantic", "hybrid"} else None
                lex = lexical.search(request) if set(methods) & {"lexical", "hybrid"} else None
                runs = {}
                if "semantic" in methods:
                    runs["semantic"] = sem
                if "lexical" in methods:
                    runs["lexical"] = lex
                if "hybrid" in methods:
                    for name, config in configs.items():
                        effective = config.model_copy(
                            update={"candidate_limit": max(config.candidate_limit, k)}
                        )
                        runs[name] = fuse_results(sem, lex, effective, depth)
                if compare_reranking:
                    pool_size = max(settings.reranking_candidate_limit, k)
                    for name, response in list(runs.items()):
                        runs[name + "_reranked"] = rerank_results(
                            response,
                            request.model_copy(update={"limit": k, "rerank": True}),
                            sessions,
                            reranker,
                            pool_size,
                        )
                relevance = {int(id_): grade for id_, grade in case["relevance"].items()}
                cases.append(
                    {
                        "id": case["id"],
                        "query": case["q"],
                        "relevance": relevance,
                        "runs": {
                            name: {
                                "metrics": retrieval_metrics(
                                    [hit.article_id for hit in response.results], relevance, k
                                ),
                                "response": response.model_dump(mode="json"),
                            }
                            for name, response in runs.items()
                        },
                    }
                )
    finally:
        if client is not None:
            client.close()
    names = list(cases[0]["runs"])
    means = {
        name: {
            metric: sum(case["runs"][name]["metrics"][metric] for case in cases) / len(cases)
            for metric in ("ndcg", "recall", "reciprocal_rank")
        }
        for name in names
    }
    wins = (
        {
            name: [
                case["id"]
                for case in cases
                if all(
                    case["runs"][name]["metrics"]["ndcg"]
                    > case["runs"][baseline]["metrics"]["ndcg"] + 1e-12
                    for baseline in ("semantic", "lexical")
                )
            ]
            for name in configs
            if name in names
        }
        if {"semantic", "lexical"} <= set(names)
        else {}
    )
    return {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset["name"],
        "dataset_hash": digest(dataset),
        "dataset_note": dataset["note"],
        "document_count": len(dataset["documents"]),
        "query_count": len(dataset["queries"]),
        "methods": list(dict.fromkeys(methods)),
        "versions": {
            name: version(name)
            for name in ("newslens-ai", "sentence-transformers", "qdrant-client", "SQLAlchemy")
        },
        "model_version": provider.version if provider else None,
        "cutoff": k,
        "reranking_model_version": reranker.version if reranker else None,
        "reranking_candidate_limit": max(settings.reranking_candidate_limit, k)
        if reranker
        else None,
        "reranking_ndcg_deltas": {
            name: means[name + "_reranked"]["ndcg"] - means[name]["ndcg"]
            for name in names
            if name + "_reranked" in means
        },
        "configs": {name: config.model_dump() for name, config in configs.items()},
        "mean_metrics": means,
        "hybrid_ndcg_wins_over_both": wins,
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on an isolated authored corpus"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", action="append", choices=["semantic", "lexical", "hybrid"])
    parser.add_argument("--compare-configs", action="store_true")
    parser.add_argument("--compare-reranking", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args(argv)
    if not 1 <= args.k <= 50:
        parser.error("Use cutoff 1..50")
    try:
        settings = Settings()
        configs = {
            "hybrid": FusionConfig(
                candidate_limit=settings.hybrid_candidate_limit,
                rrf_k=settings.hybrid_rrf_k,
                semantic_weight=settings.hybrid_semantic_weight,
                lexical_weight=settings.hybrid_lexical_weight,
            )
        }
        if args.compare_configs:
            configs.update(
                {
                    "rrf_k10": FusionConfig(rrf_k=10),
                    "semantic_weight2": FusionConfig(semantic_weight=2),
                    "lexical_weight2": FusionConfig(lexical_weight=2),
                }
            )
        report = evaluate(
            json.loads(args.dataset.read_text(encoding="utf-8")),
            settings,
            configs,
            args.method or ["semantic", "lexical", "hybrid"],
            args.k,
            compare_reranking=args.compare_reranking,
        )
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error("search_evaluation_failed error=%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
