from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime


def clean_publication_date(
    value: datetime | None, retrieved_at: datetime, *, raw_date: str | None = None
) -> tuple[datetime | None, str | None]:
    """Do not invent dates. Allow one day of clock skew; flag pre-1900 dates."""
    if raw_date is not None:
        # Revalidate the publisher string: feedparser can roll impossible dates forward.
        try:
            value = datetime.fromisoformat(raw_date.strip().replace("Z", "+00:00"))
        except ValueError:
            try:
                value = parsedate_to_datetime(raw_date)
            except (ValueError, TypeError, OverflowError):
                return None, "invalid_publication_date"
    if value is None:
        return None, None
    if value.tzinfo is None or value.utcoffset() is None:
        return None, "ambiguous_publication_timezone"
    value = value.astimezone(UTC)
    if value.year < 1900 or value > retrieved_at.astimezone(UTC) + timedelta(days=1):
        return None, "implausible_publication_date"
    return value, None
