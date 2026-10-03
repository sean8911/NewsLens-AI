import json
from pathlib import Path
from urllib.parse import urlsplit

import soupsieve
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourcePolicy(BaseModel):
    """Operator-reviewed permission, separate from publisher robots directives."""

    model_config = ConfigDict(extra="forbid")
    host: str
    terms_allow_extraction: bool = False
    terms_url: str = Field(min_length=1)
    review_note: str = Field(min_length=1)
    article_selector: str = "article"
    delay_seconds: float = Field(default=2, ge=1, le=60)

    @field_validator("host")
    @classmethod
    def exact_host(cls, value: str) -> str:
        value = value.lower()
        parsed = urlsplit("https://" + value)
        if parsed.hostname != value or parsed.port is not None or "/" in value or "*" in value:
            raise ValueError("Use an exact hostname, without scheme, path, port or wildcard")
        return value

    @field_validator("article_selector")
    @classmethod
    def valid_selector(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Article selector cannot be empty")
        try:
            soupsieve.compile(value)
        except Exception as exc:
            raise ValueError("Invalid article CSS selector") from exc
        return value


def load_policies(path: Path | None) -> dict[str, SourcePolicy]:
    if path is None:
        return {}
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("Content policies must be a JSON list")
    policies = [SourcePolicy.model_validate(value) for value in values]
    if len({policy.host for policy in policies}) != len(policies):
        raise ValueError("Duplicate source policy host")
    return {policy.host: policy for policy in policies}
