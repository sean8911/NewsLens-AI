"""Bounded, process-local protection for the single-worker native API."""

import logging
import math
import time
from threading import Lock

import anyio
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)
INFERENCE_PATHS = {"/search", "/chat", "/research"}


class RequestBudget:
    def __init__(self, settings, clock=time.monotonic):
        self.settings = settings
        self.clock = clock
        self.clients = {}
        self.inflight = 0
        self.lock = Lock()

    def admit(self, client, inference):
        with self.lock:
            now = self.clock()
            self.clients = {key: value for key, value in self.clients.items() if value[0] > now}
            if client not in self.clients:
                if len(self.clients) >= self.settings.api_max_clients:
                    return 429, 60
                self.clients[client] = [now + 60, 0, 0]
            bucket = self.clients[client]
            if (
                bucket[1] >= self.settings.api_requests_per_minute
                or inference
                and bucket[2] >= self.settings.api_inference_requests_per_minute
            ):
                return 429, max(1, math.ceil(bucket[0] - now))
            bucket[1] += 1
            if inference:
                bucket[2] += 1
                if self.inflight >= self.settings.api_max_inflight_inference:
                    return 503, 1
                self.inflight += 1
            return None

    def release(self, inference):
        if inference:
            with self.lock:
                self.inflight -= 1


async def validation_error(request, exc: RequestValidationError):
    # Pydantic's default response echoes rejected values and custom exception context.
    # Even an extra-field location can be attacker-controlled. Return only stable types.
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request",
            "errors": [{"type": item["type"]} for item in exc.errors()],
        },
    )


class APISecurityMiddleware:
    def __init__(self, app, settings):
        self.app = app
        self.settings = settings
        self.budget = RequestBudget(settings)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = False

        async def safe_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = [
                    item
                    for item in message.get("headers", [])
                    if item[0].lower() not in {b"cache-control", b"x-content-type-options"}
                ]
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"cache-control", b"no-store"),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        async def reject(status, detail, retry=None):
            headers = {"Retry-After": str(retry)} if retry is not None else None
            await JSONResponse({"detail": detail}, status_code=status, headers=headers)(
                scope, receive, safe_send
            )

        # Health stays available during load and never opens database/model resources.
        inference = scope["path"].rstrip("/") in INFERENCE_PATHS
        admitted = scope["path"] != "/health"
        if admitted:
            # Do not trust client-controlled X-Forwarded-For / Forwarded headers.
            client = (scope.get("client") or ("unknown", 0))[0]
            refused = self.budget.admit(client, inference)
            if refused:
                status, retry = refused
                return await reject(
                    status,
                    "Rate limit exceeded"
                    if status == 429
                    else "Inference capacity busy. Try again later.",
                    retry,
                )
        try:
            if len(scope.get("query_string", b"")) > 16384:
                return await reject(414, "Query too large")
            headers = dict(scope.get("headers", []))
            length = headers.get(b"content-length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError:
                    return await reject(400, "Invalid content length")
                if declared < 0:
                    return await reject(400, "Invalid content length")
                if declared > self.settings.api_max_body_bytes:
                    return await reject(413, "Request too large")
            body = bytearray()
            try:
                with anyio.fail_after(self.settings.api_body_timeout_seconds):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        chunk = message.get("body", b"")
                        if len(body) + len(chunk) > self.settings.api_max_body_bytes:
                            return await reject(413, "Request too large")
                        body.extend(chunk)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await reject(408, "Request body timed out")

            async def buffered_receive():
                nonlocal body
                if body is not None:
                    data, body = bytes(body), None
                    return {"type": "http.request", "body": data, "more_body": False}
                return await receive()

            try:
                await self.app(scope, buffered_receive, safe_send)
            except Exception as exc:
                # Never log exception text, request bodies, URLs, credentials or tracebacks.
                logger.error("api_unhandled_error type=%s", type(exc).__name__)
                if not started:
                    await reject(500, "Internal service error")
                else:
                    raise
        finally:
            if admitted:
                self.budget.release(inference)
