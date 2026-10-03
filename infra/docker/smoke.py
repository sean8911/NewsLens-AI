"""Run inside the backend container after Compose reports healthy services."""

import json
from urllib.request import urlopen


def read(url):
    with urlopen(url, timeout=15) as response:
        assert response.status == 200, url
        return response.read()


def main():
    checks = []
    api = json.loads(read("http://backend:8000/health"))
    assert api["status"] == "ok"
    checks.append("backend health")
    status = json.loads(read("http://backend:8000/system/status"))
    assert status["database"] == "available" and status["article_count"] >= 0
    checks.append("migrated PostgreSQL catalog")
    read("http://qdrant:6333/readyz")
    checks.append("Qdrant readiness")
    assert "collections" in json.loads(read("http://qdrant:6333/collections"))["result"]
    checks.append("Qdrant collections API")
    proxy = json.loads(read("http://frontend:3000/api/health"))
    assert proxy == api
    checks.append("frontend to backend proxy")
    assert b"NewsLens" in read("http://frontend:3000/chat")
    checks.append("frontend chat page")
    print(json.dumps({"passed": True, "checks": checks, "version": api["version"]}))


if __name__ == "__main__":
    main()
