import json
from pathlib import Path
from time import perf_counter

from nlp.classification.classifier import HYPOTHESES, TopicClassifier


def evaluate(path: Path, classifier: TopicClassifier) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation requires a nonempty JSON list")
    tp = fp = fn = errors = 0
    results = []
    started = perf_counter()
    for case in cases:
        expected = set(case["expected_topics"])
        if not expected or not expected <= HYPOTHESES.keys():
            raise ValueError("Unknown or empty expected topic set")
        prediction = classifier.classify(
            case["title"], case.get("description"), case.get("content"), case.get("language")
        )
        actual = {label.topic for label in prediction.article_topics}
        tp += len(expected & actual)
        fp += len(actual - expected)
        fn += len(expected - actual)
        errors += prediction.topic_status in {"model_unavailable", "error"}
        results.append(
            {"id": case["id"], "expected": sorted(expected), "prediction": prediction.values()}
        )
    return {
        "model": classifier.version,
        "cases": len(cases),
        "errors": errors,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else 0,
        "micro_recall": tp / (tp + fn) if tp + fn else 0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0,
        "duration_seconds": perf_counter() - started,
        "results": results,
    }
