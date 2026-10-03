import pytest

from backend.app.core.config import Settings
from backend.app.database.session import DatabaseConfigurationError, database_url
from ingestion.cli import main


@pytest.mark.parametrize("value", [None, "not a URL", "sqlite:///db.sqlite", "postgresql://host/"])
def test_invalid_database_configuration(value):
    with pytest.raises(DatabaseConfigurationError):
        database_url(Settings(_env_file=None, database_url=value))


def test_standard_postgres_url_uses_pg8000_and_masks_password():
    settings = Settings(
        _env_file=None, database_url="postgresql://user:example-secret@localhost/newslens"
    )
    url = database_url(settings)
    assert url.drivername == "postgresql+pg8000"
    assert "example-secret" not in repr(settings)
    assert "example-secret" not in str(url)


def test_store_requires_configuration_before_fetching(monkeypatch):
    from ingestion.http import RSSFetcher

    def fail(*args):
        pytest.fail("Missing database configuration must fail before fetching")

    monkeypatch.setattr(RSSFetcher, "fetch", fail)
    with pytest.raises(SystemExit) as exc:
        main(["--store"])
    assert exc.value.code == 2


def test_remote_database_tls_requires_verified_certificate_and_hostname(monkeypatch):
    import ssl

    from backend.app.database import session

    captured = {}
    monkeypatch.setattr(session, "create_engine", lambda *args, **kwargs: captured.update(kwargs))
    session.create_database_engine(
        Settings(
            _env_file=None,
            database_url="postgresql://user:secret@db.example/test",
            database_ssl=True,
        )
    )
    context = captured["connect_args"]["ssl_context"]
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
    assert captured["hide_parameters"]


@pytest.mark.parametrize(
    "values",
    [
        {"database_ssl_ca_file": "private-missing-ca.pem"},
        {"database_ssl": True, "database_ssl_ca_file": "private-missing-ca.pem"},
        {"database_url": "postgresql://user:private-secret@db.example/test?sslmode=require"},
    ],
)
def test_invalid_tls_configuration_fails_without_exposing_credentials(values):
    from backend.app.database.session import create_database_engine

    config = {"database_url": "postgresql://user:secret@db.example/test", **values}
    with pytest.raises(DatabaseConfigurationError) as exc:
        create_database_engine(Settings(_env_file=None, **config))
    assert "private-secret" not in str(exc.value)
    assert "private-missing-ca" not in str(exc.value)
