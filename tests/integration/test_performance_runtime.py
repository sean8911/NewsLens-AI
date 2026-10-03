import json
import logging

import pytest

from backend.app.core import performance as perf
from backend.app.schemas.search import SearchRequest
from backend.app.search_runtime import search_snapshot

pytestmark = pytest.mark.integration


def test_search_timer_excludes_snapshot_consumer(database, monkeypatch):
    rows = []

    class Capture(logging.Handler):
        def emit(self, record):
            rows.append(json.loads(record.getMessage()))

    monkeypatch.setattr(perf.logger, "handlers", [Capture()])
    tick = [0.0]
    monkeypatch.setattr(perf, "perf_counter", lambda: tick[0])
    with perf.operation("chat"):
        with search_snapshot(SearchRequest(q="unmatched", method="lexical"), database[2]):
            tick[0] = 100  # Represents evidence/generation in the caller, not search work.
    assert len(rows) == 1
    assert rows[0]["search_time_ms"] == 0
    assert rows[0]["total_time_ms"] == 100000
    assert rows[0]["stage_calls"]["retrieval"] == 1
    assert rows[0]["stage_calls"]["embedding"] == 0
