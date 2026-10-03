import calendar
import re
from datetime import UTC, date, datetime, timedelta

from backend.app.schemas.search import SearchRequest
from backend.app.schemas.temporal import TimeIntent

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_PATTERN = "|".join(MONTHS)
MONTH = re.compile(rf"\b(?:in|during|for)\s+({MONTH_PATTERN})(?:\s+(20\d{{2}}|19\d{{2}}))?\b", re.I)
LATEST = re.compile(
    r"\b(?:latest|newest|recent(?:ly)?|up[- ]to[- ]date|right now|terbaru|terkini)\b"
    r"|最近|最新|近期",
    re.I,
)


def resolve_time_intent(request: SearchRequest, now: datetime) -> TimeIntent:
    if now.tzinfo is None:
        raise ValueError("Time intent requires an aware UTC reference")
    now = now.astimezone(UTC)
    today = now.date()
    base = dict(
        as_of=now, retrieval_query=request.q, date_from=request.date_from, date_to=request.date_to
    )
    if request.recency == "off":
        return TimeIntent(kind="none", reason="disabled", **base)
    if request.date_from or request.date_to:
        return TimeIntent(kind="explicit", reason="explicit_filters_take_precedence", **base)

    def resolved(
        kind: str,
        match,
        start: date | None = None,
        end: date | None = None,
        reason: str = "detected",
    ) -> TimeIntent:
        query = " ".join((request.q[: match.start()] + " " + request.q[match.end() :]).split())
        if not any(char.isalnum() for char in query):
            query = request.q
        return TimeIntent(
            kind=kind,
            as_of=now,
            retrieval_query=query,
            date_from=start,
            date_to=end,
            expression=match.group(),
            reason=reason,
        )

    # Multiple periods/comparisons need explicit filters; never silently choose one side.
    if (
        len(re.findall(rf"\b(?:{MONTH_PATTERN})\b", request.q, re.I)) > 1
        or len(re.findall(r"\b(?:19|20)\d{2}\b", request.q)) > 1
        or re.search(r"\b(?:between|versus|vs\.?|compare)\b", request.q, re.I)
    ):
        return TimeIntent(kind="none", reason="ambiguous_or_comparative_time", **base)
    match = MONTH.search(request.q)
    if match:
        month = MONTHS[match[1].lower()]
        year = int(match[2]) if match[2] else today.year - (month > today.month)
        return resolved(
            "historical",
            match,
            date(year, month, 1),
            date(year, month, calendar.monthrange(year, month)[1]),
            "named_month" if match[2] else "most_recent_month_occurrence",
        )
    match = re.search(r"\b(?:on|in|during)\s+(\d{4}-\d{2}-\d{2})\b", request.q, re.I)
    if match:
        try:
            day = date.fromisoformat(match[1])
        except ValueError:
            return TimeIntent(kind="none", reason="unresolved_invalid_date", **base)
        return resolved("historical", match, day, day)
    match = re.search(r"\b(?:in|during)\s+((?:19|20)\d{2})\b", request.q, re.I)
    if match:
        year = int(match[1])
        return resolved("historical", match, date(year, 1, 1), date(year, 12, 31))
    match = re.search(
        r"\b(?:last month|yesterday|today|this week|this month|last week)\b", request.q, re.I
    )
    if match:
        phrase = match.group().lower()
        monday = today - timedelta(days=today.weekday())
        if phrase == "last month":
            end = today.replace(day=1) - timedelta(days=1)
            return resolved("historical", match, end.replace(day=1), end)
        if phrase == "last week":
            return resolved(
                "historical", match, monday - timedelta(days=7), monday - timedelta(days=1)
            )
        if phrase == "yesterday":
            return resolved(
                "historical", match, today - timedelta(days=1), today - timedelta(days=1)
            )
        start = (
            today
            if phrase == "today"
            else monday
            if phrase == "this week"
            else today.replace(day=1)
        )
        return resolved("latest", match, start, today)
    match = re.search(r"\b(?:past|last)\s+(\d{1,3})\s+days?\b", request.q, re.I)
    if match:
        days = int(match[1])
        if not 1 <= days <= 365:
            return TimeIntent(kind="none", reason="unresolved_day_count", **base)
        return resolved("latest", match, today - timedelta(days=days - 1), today)
    match = LATEST.search(request.q)
    if match:
        return resolved("latest", match)
    return TimeIntent(
        kind="latest" if request.recency == "latest" else "none",
        reason="requested_latest" if request.recency == "latest" else "no_time_intent",
        **base,
    )
