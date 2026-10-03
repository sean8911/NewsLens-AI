"""Small real-model translation matrix with explicit geography, risk and negation checks."""

import json
from pathlib import Path

from backend.app.core.config import Settings
from nlp.multilingual.local_translation import LocalTranslator
from nlp.multilingual.translation import translate_excerpt

ROOT = Path(__file__).resolve().parents[1]


def main():
    dataset = json.loads(
        (ROOT / "backend/app/evaluation/fixtures/multilingual.json").read_text("utf-8")
    )
    model = LocalTranslator(Settings())
    report = {"passed": False, "translation_model_version": model.version, "cases": []}
    try:
        for document in dataset["documents"][:3]:
            for target in ("en", "ms", "zh"):
                if target == document["language"]:
                    continue
                output, translated = translate_excerpt(document["description"], target, model)
                case = {
                    "source": document["language"],
                    "target": target,
                    "original": document["description"],
                    "translation": output,
                }
                report["cases"].append(case)
                assert translated
                expected = {
                    "en": ["Malaysia", "risk"],
                    "ms": ["Malaysia", "risiko"],
                    "zh": ["马来西亚", "风险"],
                }[target]
                assert all(term.lower() in output.lower() for term in expected), case
                print(f"PASS {document['language']} -> {target}", flush=True)
        for target, markers in (("ms", ("tidak", "belum", "bukan")), ("zh", ("不", "未", "没"))):
            original = "Malaysia did not approve the AI safety policy."
            output, _ = translate_excerpt(original, target, model)
            case = {
                "source": "en",
                "target": target,
                "original": original,
                "translation": output,
                "check": "negation",
            }
            report["cases"].append(case)
            assert any(marker in output for marker in markers), case
            print(f"PASS negation en -> {target}", flush=True)
        report["passed"] = True
    finally:
        (ROOT / "docs/phase-23-translation-matrix.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
