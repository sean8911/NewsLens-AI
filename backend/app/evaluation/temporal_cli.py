import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import text

from backend.app.core.config import Settings
from backend.app.database.session import create_session_factory
from backend.app.evaluation.cli import evaluation_database
from backend.app.evaluation.metrics import retrieval_metrics
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchRequest
from backend.app.schemas.temporal import RecencyConfig
from backend.app.search_runtime import search_snapshot
from backend.app.services.article_embeddings import ArticleEmbeddingService
from backend.app.services.recency import decay_factor
from nlp.embeddings.service import digest
from nlp.embeddings.store import open_store
from nlp.multilingual.runtime import provider_for

DATASET = Path(__file__).parent / "fixtures" / "temporal.json"


def evaluate_time(settings: Settings) -> dict:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    as_of = datetime.fromisoformat(dataset["as_of"])
    cases = []
    with (
        evaluation_database(settings) as engine,
        TemporaryDirectory(prefix="newslens_time_") as directory,
    ):
        with engine.connect() as connection:
            schema = connection.scalar(text("SELECT current_schema()"))
        isolated = settings.model_copy(
            update={
                "database_url": settings.test_database_url,
                "database_schema": schema,
                "qdrant_url": None,
                "qdrant_path": Path(directory) / "qdrant",
            }
        )
        sessions = create_session_factory(engine)
        with sessions.begin() as session:
            for document in dataset["documents"]:
                values = dict(document)
                values["published_at"] = (
                    datetime.fromisoformat(values["published_at"])
                    if values["published_at"]
                    else None
                )
                session.add(
                    ArticleRecord(
                        **values,
                        url=f"https://example.com/time/{document['id']}",
                        source="Synthetic evaluation",
                        category="Technology",
                        language="en",
                        retrieved_at=as_of,
                        cleaning_version="1",
                        processing_status="clean",
                    )
                )
        provider = provider_for(isolated)
        store = open_store(isolated, provider)
        try:
            if ArticleEmbeddingService(sessions, provider, store).sync()["errors"]:
                raise RuntimeError("Time evaluation indexing failed")
        finally:
            store.client.close()
        for case in dataset["queries"]:
            runs = {}
            for mode in ("off", "exponential", "gaussian", "hyperbolic"):
                configured = isolated.model_copy(
                    update={"recency_decay": "exponential" if mode == "off" else mode}
                )
                request = SearchRequest(
                    q=case["q"],
                    method="hybrid",
                    rerank=True,
                    limit=3,
                    recency="off" if mode == "off" else "auto",
                )
                with search_snapshot(request, configured, as_of=as_of) as (response, _):
                    runs[mode] = {
                        "response": response.model_dump(mode="json"),
                        "metrics": retrieval_metrics(
                            [hit.article_id for hit in response.results],
                            {int(key): value for key, value in case["relevance"].items()},
                            3,
                        ),
                    }
            cases.append({**case, "runs": runs})
    means = {
        mode: sum(case["runs"][mode]["metrics"]["ndcg"] for case in cases) / len(cases)
        for mode in ("off", "exponential", "gaussian", "hyperbolic")
    }
    return {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset["name"],
        "dataset_hash": digest(dataset),
        "note": dataset["note"],
        "as_of": dataset["as_of"],
        "document_count": len(dataset["documents"]),
        "cutoff": 3,
        "mean_ndcg": means,
        "decay_curves": {
            mode: {
                str(age): decay_factor(
                    age, RecencyConfig(decay=mode, half_life_days=settings.recency_half_life_days)
                )
                for age in (0, 1, 7, 14, 30, 90)
            }
            for mode in ("exponential", "gaussian", "hyperbolic")
        },
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare recency decay on isolated dated news")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_time(Settings())
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error("time_evaluation_failed error=%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
