import ipaddress
import socket
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit

import requests
from protego import Protego

from backend.app.core.config import Settings
from ingestion.extraction.policy import SourcePolicy

USER_AGENT = "NewsLensAI/0.5 (article extraction)"
# Shared within a worker process so a new collection cannot reset publisher backoff.
_NEXT_REQUEST: dict[str, float] = {}


class ExtractionBlocked(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ArticleFetcher:
    """Sequential bounded requests. Never authenticates, executes JS, or bypasses barriers."""

    def __init__(self, settings: Settings, policies: dict[str, SourcePolicy]) -> None:
        self.settings = settings
        self.policies = policies
        self.session = requests.Session()
        self.session.trust_env = False  # Do not pick up .netrc credentials or implicit proxies.
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.robots: dict[str, Protego | None] = {}
        self.next_request = _NEXT_REQUEST
        self.blocked_hosts: set[str] = set()

    def validate(self, url: str) -> tuple[str, SourcePolicy]:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        policy = self.policies.get(host)
        if not policy or not policy.terms_allow_extraction:
            raise ExtractionBlocked("policy_denied")
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 443)
        ):
            raise ExtractionBlocked("unsafe_url")
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        if not addresses or any(
            not ipaddress.ip_address(item[4][0]).is_global for item in addresses
        ):
            raise ExtractionBlocked("unsafe_url")
        return host, policy

    def request(self, url: str, delay: float, *, robots: bool = False) -> bytes:
        host, _ = self.validate(url)
        if host in self.blocked_hosts:
            raise ExtractionBlocked("host_backoff")
        wait = self.next_request.get(host, 0) - time.monotonic()
        if wait > 60:
            raise ExtractionBlocked("rate_deferred")
        if wait > 0:
            time.sleep(wait)
        try:
            # Redirects are handled by fetch(), with policy + robots checks at every hop.
            with self.session.get(
                url,
                allow_redirects=False,
                stream=True,
                timeout=(self.settings.connect_timeout_seconds, self.settings.read_timeout_seconds),
                headers={"Accept": "text/plain" if robots else "text/html, application/xhtml+xml"},
            ) as response:
                if response.status_code in {401, 403, 429, 503}:
                    self.blocked_hosts.add(host)  # No further requests to this host this run.
                    retry_after = response.headers.get("Retry-After", "")
                    cooldown = 3600.0
                    try:
                        cooldown = max(cooldown, float(retry_after))
                    except ValueError:
                        try:
                            cooldown = max(
                                cooldown,
                                parsedate_to_datetime(retry_after).timestamp() - time.time(),
                            )
                        except (ValueError, TypeError, OverflowError):
                            pass
                    self.next_request[host] = time.monotonic() + cooldown
                    raise ExtractionBlocked("http_restricted")
                if 300 <= response.status_code < 400:
                    if robots:
                        raise ExtractionBlocked("robots_unavailable")
                    raise PageRedirect(urljoin(url, response.headers.get("Location", "")))
                if response.status_code != 200:
                    raise ExtractionBlocked("robots_unavailable" if robots else "http_error")
                mime = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                allowed = {"text/plain"} if robots else {"text/html", "application/xhtml+xml"}
                if mime not in allowed:
                    raise ExtractionBlocked("unsupported_type")
                chunks, size = [], 0
                limit = 512_000 if robots else self.settings.content_max_bytes
                started = time.monotonic()
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > limit:
                        raise ExtractionBlocked("too_large")
                    if time.monotonic() - started > 30:
                        raise ExtractionBlocked("download_timeout")
                    chunks.append(chunk)
                return b"".join(chunks)
        finally:
            # Start spacing at request completion, including errors and redirects.
            self.next_request[host] = max(self.next_request.get(host, 0), time.monotonic() + delay)

    def fetch(self, url: str) -> bytes:
        for _ in range(6):
            host, policy = self.validate(url)
            origin = "https://" + host
            if origin not in self.robots:
                self.robots[origin] = None  # Fail closed and cache failures for this run.
                payload = self.request(origin + "/robots.txt", policy.delay_seconds, robots=True)
                decoded = payload.decode("utf-8-sig", errors="strict")
                lines = [line.split("#", 1)[0].strip() for line in decoded.splitlines()]
                if any(lines) and not any(line.lower().startswith("user-agent:") for line in lines):
                    raise ExtractionBlocked("robots_unavailable")
                self.robots[origin] = Protego.parse(decoded)
            rules = self.robots[origin]
            if rules is None:
                raise ExtractionBlocked("robots_unavailable")
            if not rules.can_fetch(url, USER_AGENT):
                raise ExtractionBlocked("robots_denied")
            delay = max(policy.delay_seconds, rules.crawl_delay(USER_AGENT) or 0)
            rate = rules.request_rate(USER_AGENT)
            if rate:
                if rate.start_time is not None or rate.end_time is not None or rate.requests <= 0:
                    raise ExtractionBlocked("unsupported_rate")
                delay = max(delay, rate.seconds / rate.requests)
            if delay > 60:
                raise ExtractionBlocked("crawl_delay_exceeded")
            # Apply newly discovered robots delay to the first article request as well.
            self.next_request[host] = max(self.next_request.get(host, 0), time.monotonic() + delay)
            try:
                return self.request(url, delay)
            except PageRedirect as redirect:
                if urlsplit(redirect.url).hostname != host:
                    raise ExtractionBlocked("redirect_host_denied") from redirect
                url = redirect.url
        raise ExtractionBlocked("redirect_limit")

    def close(self) -> None:
        self.session.close()


class PageRedirect(Exception):
    def __init__(self, url: str) -> None:
        self.url = url
