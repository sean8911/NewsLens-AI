from urllib.parse import unquote_plus, urlsplit, urlunsplit

from pydantic import HttpUrl, TypeAdapter

TRACKING_KEYS = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid"}
BBC_KEYS = {"at_campaign", "at_medium", "at_link_id", "at_link_origin", "at_link_type"}
SIGNED_KEYS = {"signature", "sig", "token", "expires"}


def clean_url(value: str) -> str:
    """Remove only known tracking keys, preserving all other query bytes and fragments."""
    if any(ord(char) < 32 for char in value):
        raise ValueError("URL contains control characters")
    url = str(TypeAdapter(HttpUrl).validate_python(value))
    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        raise ValueError("Article URLs must not contain credentials")
    pairs = parts.query.split("&") if parts.query else []
    keys = [unquote_plus(pair.partition("=")[0]).casefold() for pair in pairs]
    if any(key in SIGNED_KEYS or key.startswith(("x-amz-", "x-goog-")) for key in keys):
        return url
    host = parts.hostname or ""
    bbc = any(host == domain or host.endswith("." + domain) for domain in ("bbc.com", "bbc.co.uk"))
    kept = [
        pair
        for pair, key in zip(pairs, keys, strict=True)
        if not key.startswith("utm_") and key not in TRACKING_KEYS and not (bbc and key in BBC_KEYS)
    ]
    return urlunsplit(parts._replace(query="&".join(kept)))
