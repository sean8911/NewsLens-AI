"""Validate rendered Compose wiring without requiring a running Docker Engine."""

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    docker = shutil.which("docker")
    standalone = ROOT / ".local/tools/docker-compose.exe"
    cli = [docker, "compose"] if docker else [str(standalone)]
    if not docker and not standalone.is_file():
        raise SystemExit("Install Docker Compose before validating the configuration")
    checks = []
    for variant, files in [
        ("base", ["compose.yaml"]),
        ("development", ["compose.yaml", "compose.dev.yaml"]),
        ("existing-models", ["compose.yaml", "compose.models.yaml"]),
        ("combined", ["compose.yaml", "compose.dev.yaml", "compose.models.yaml"]),
    ]:
        args = [item for name in files for item in ["-f", name]]
        result = subprocess.run(
            cli
            + args
            + ["--profile", "worker", "--profile", "tools", "config", "--format", "json"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        services = json.loads(result.stdout)["services"]
        assert set(services) == {
            "postgres",
            "qdrant",
            "migrate",
            "backend",
            "frontend",
            "worker",
            "models",
        }
        assert (
            services["backend"]["depends_on"]["migrate"]["condition"]
            == "service_completed_successfully"
        )
        assert services["frontend"]["depends_on"]["backend"]["condition"] == "service_healthy"
        assert services["migrate"]["depends_on"]["postgres"]["condition"] == "service_healthy"
        assert services["backend"]["environment"]["NEWSLENS_QDRANT_URL"] == "http://qdrant:6333"
        assert services["frontend"]["environment"]["NEWSLENS_API_URL"] == "http://backend:8000"
        for name, service in services.items():
            for port in service.get("ports", []):
                assert name in {"backend", "frontend"} and port["host_ip"] == "127.0.0.1"
        assert services["worker"]["profiles"] == ["worker"]
        assert services["models"]["profiles"] == ["tools"]
        if "compose.models.yaml" in files:
            mount = next(v for v in services["backend"]["volumes"] if v["target"] == "/models")
            assert mount["type"] == "bind" and mount["read_only"]
            assert not mount["bind"]["create_host_path"]
        if "compose.dev.yaml" in files:
            assert "--reload" in services["backend"]["command"]
            assert services["frontend"]["build"]["target"] == "development"
        checks.append(variant)
    report = {
        "passed": True,
        "validated_variants": checks,
        "scope": "Compose parser and rendered wiring only; no image build or container execution",
    }
    (ROOT / "docs/phase-35-compose-validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Passed {len(checks)} rendered Compose variants")


if __name__ == "__main__":
    main()
