import json
from pathlib import Path
from time import perf_counter

from nlp.entities.extractor import ENTITY_TYPES, EntityExtractor, canonical_name


def evaluate(path: Path, extractor: EntityExtractor) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation requires a nonempty JSON list")
    tp = fp = fn = errors = 0
    results = []
    started = perf_counter()
    for case in cases:
        expected = set()
        for item in case["expected_entities"]:
            if item["entity_type"] not in ENTITY_TYPES or not item["name"].strip():
                raise ValueError("Invalid expected entity")
            expected.add((item["entity_type"], canonical_name(item["name"], item["entity_type"])))
        prediction = extractor.extract(
            case["title"], case.get("description"), case.get("content"), case.get("language")
        )
        actual = {(entity.entity_type, entity.canonical_name) for entity in prediction.entities}
        tp += len(expected & actual)
        fp += len(actual - expected)
        fn += len(expected - actual)
        errors += prediction.entity_status in {"model_unavailable", "error"}
        results.append(
            {"id": case["id"], "expected": sorted(expected), "prediction": prediction.values()}
        )
    return {
        "model": extractor.version,
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
