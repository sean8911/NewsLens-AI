"""Content-free, per-operation telemetry for synchronous application workers."""

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from time import perf_counter
from uuid import uuid4

STAGES = (
    "ingestion",
    "nlp",
    "embedding",
    "search",
    "retrieval",
    "reranking",
    "generation",
    "chatbot",
)
_current: ContextVar = ContextVar("newslens_performance", default=None)
logger = logging.getLogger("newslens.performance")
# Dedicated JSONL on stderr works under both the CLI and `uvicorn ...:app`.
# It never alters application log levels or writes prompts to stdout.
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


@dataclass
class Measurement:
    operation: str
    operation_id: str = field(default_factory=lambda: uuid4().hex)
    outcome: str = "succeeded"
    error_type: str | None = None
    elapsed: dict = field(default_factory=lambda: dict.fromkeys(STAGES, 0.0))
    calls: dict = field(default_factory=lambda: dict.fromkeys(STAGES, 0))
    failures: dict = field(default_factory=lambda: dict.fromkeys(STAGES, 0))
    depth: dict = field(default_factory=lambda: dict.fromkeys(STAGES, 0))
    usage: dict = field(
        default_factory=lambda: {
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "embedding_input_tokens": 0,
            "llm_calls": 0,
            "embedding_calls": 0,
            "llm_completed_calls": 0,
            "embedding_completed_calls": 0,
        }
    )

    def event(self, total_ms: float) -> dict:
        return {
            "event": "performance",
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "operation": self.operation,
            "operation_id": self.operation_id,
            "outcome": self.outcome,
            "error_type": self.error_type,
            **{f"{name}_time_ms": round(value, 3) for name, value in self.elapsed.items()},
            "total_time_ms": round(total_ms, 3),
            "stage_calls": self.calls.copy(),
            "stage_failures": self.failures.copy(),
            "usage": {
                **self.usage,
                "token_method": "local_tokenizer_observed",
                "coverage": "instrumented_local_model_calls_only",
                "llm_api_cost_usd": 0.0,
                "embedding_api_cost_usd": 0.0,
                "cost_basis": "local_inference_no_external_api_charges",
                "hardware_cost_usd": None,
            },
        }


@contextmanager
def operation(name: str):
    """Nested operations share one record; every top-level call emits, even on failure."""
    active = _current.get()
    if active is not None:
        yield active
        return
    measurement = Measurement(name)
    token = _current.set(measurement)
    started = perf_counter()
    try:
        yield measurement
    except BaseException as exc:
        measurement.outcome = "interrupted" if not isinstance(exc, Exception) else "failed"
        measurement.error_type = type(exc).__name__
        raise
    finally:
        total_ms = (perf_counter() - started) * 1000
        _current.reset(token)
        # An unavailable log sink must not turn a completed transaction into a failure.
        try:
            logger.info(json.dumps(measurement.event(total_ms), allow_nan=False))
        except Exception:
            pass


@contextmanager
def stage(name: str):
    """Inclusive stage time; nested occurrences of the SAME stage count only once."""
    if name not in STAGES:
        raise ValueError("Unknown performance stage")
    with operation(name) as measurement:
        outermost = measurement.depth[name] == 0
        measurement.depth[name] += 1
        started = perf_counter()
        if outermost:
            measurement.calls[name] += 1
        try:
            yield
        except BaseException:
            if outermost:
                measurement.failures[name] += 1
            raise
        finally:
            measurement.depth[name] -= 1
            if outermost:
                measurement.elapsed[name] += (perf_counter() - started) * 1000


def measured(name: str):
    """Time an ordinary synchronous function (not a generator/coroutine)."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def monitored(name: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with operation(name) as measurement:
                result = function(*args, **kwargs)
                # Fixed application enums only; never serialize arbitrary result values.
                status = getattr(result, "status", None)
                report = getattr(result, "report", None)
                if report is not None:
                    status = report.status
                if status in {
                    "answered",
                    "insufficient_information",
                    "needs_clarification",
                    "succeeded",
                    "partial",
                    "failed",
                    "interrupted",
                }:
                    measurement.outcome = status
                if getattr(result, "error_type", None) is not None:
                    measurement.outcome = "failed"
                return result

        return wrapped

    return decorate


def record_tokens(
    kind: str, input_tokens: int = 0, output_tokens: int = 0, *, calls: int = 0, completed: int = 0
) -> None:
    """Record actual tokenizer counts; input can be known even if generation fails."""
    if kind not in {"llm", "embedding"} or min(input_tokens, output_tokens, calls, completed) < 0:
        raise ValueError("Invalid token usage")
    active = _current.get()
    if active is None:
        return
    active.usage[f"{kind}_input_tokens"] += int(input_tokens)
    active.usage[f"{kind}_calls"] += calls
    active.usage[f"{kind}_completed_calls"] += completed
    if kind == "llm":
        active.usage["llm_output_tokens"] += int(output_tokens)
