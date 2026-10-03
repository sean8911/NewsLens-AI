import argparse
import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.event_runtime import config_for
from nlp.embeddings.cli import provider_for
from nlp.embeddings.service import prepare
from nlp.events.clustering import EventDocument, cluster, compare

DATASET = Path(__file__).parent / "fixtures/events.json"


def load_cases(provider):
    dataset = json.loads(DATASET.read_text("utf-8"))
    docs = [
        EventDocument(
            row["id"],
            row["title"],
            datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
            provider.embed(prepare(row["title"], row["description"], None).text),
            "en",
        )
        for row in dataset["documents"]
    ]
    return dataset, docs


def evaluate(settings):
    provider = provider_for(settings)
    dataset, docs = load_cases(provider)
    config = config_for(settings)
    labels = {row["id"]: row["event"] for row in dataset["documents"]}
    expected = {tuple(sorted((a, b))) for a, b in combinations(labels, 2) if labels[a] == labels[b]}
    runs = {}
    by_id = {doc.article_id: doc for doc in docs}
    for method in ("agglomerative", "dbscan", "hdbscan"):
        groups = cluster(docs, config, method)
        predicted = {tuple(sorted(pair)) for group in groups for pair in combinations(group, 2)}
        tp, fp, fn = len(predicted & expected), len(predicted - expected), len(expected - predicted)
        violations = [
            compare(by_id[a], by_id[b], config)
            for a, b in sorted(predicted)
            if not compare(by_id[a], by_id[b], config)["compatible"]
        ]
        runs[method] = {
            "groups": groups,
            "true_positive_pairs": tp,
            "false_positive_pairs": fp,
            "false_negative_pairs": fn,
            "pair_precision": tp / (tp + fp) if tp + fp else 0,
            "pair_recall": tp / (tp + fn) if tp + fn else 0,
            "pair_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "constraint_violations": violations,
        }
    return {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "model_version": provider.version,
        "policy": config.model_dump(),
        "runs": runs,
        "pairs": [compare(a, b, config) for a, b in combinations(docs, 2)],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare three real-model event clustering methods"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(Settings())
    args.output.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps(result["runs"], indent=2))


if __name__ == "__main__":
    main()
