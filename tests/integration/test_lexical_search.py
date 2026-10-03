from datetime import UTC, datetime

import pytest
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select, text

from backend.app.database.session import create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchRequest
from backend.app.services.lexical_search import LexicalSearchService

pytestmark = pytest.mark.integration


@pytest.fixture
def lexical_setup(database):
    sessions = create_session_factory(database[0])
    with sessions.begin() as session:
        records = []
        for i, (title, description, content) in enumerate(
            [
                (
                    "Ada Lovelace presents GPT-6-A12",
                    "Acme Robotics introduces the Nova Phone X.",
                    "Shares of $NVDA and BRK.B rose after the launch. "
                    "The new GPT-6-A12 was demonstrated.",
                ),
                (
                    "Ada Byron discusses GPT-6-B12",
                    "Different hardware from Another Company.",
                    "NVDAX shares fell. Engineers tested GPT-6-A12X.",
                ),
                (
                    "A launch report",
                    "Acme Robotics unveils another device.",
                    "NVDA trading volumes rose.",
                ),
                ("Ada", "Lovelace", None),
            ]
        ):
            record = ArticleRecord(
                title=title,
                description=description,
                content=content,
                source="Example" if i != 2 else "Other",
                category="Technology" if i != 2 else "Business",
                language="en" if i != 2 else None,
                published_at=datetime(2026, 9, 9, 23, 59, 59, 999999, UTC) if i != 2 else None,
                retrieved_at=datetime.now(UTC),
                url=f"https://example.com/lexical/{i}",
                cleaning_version="1",
                processing_status="clean",
            )
            session.add(record)
            records.append(record)
    return sessions, LexicalSearchService(sessions), records


def search(service, q, **kwargs):
    return service.search(SearchRequest(q=q, method="lexical", **kwargs))


@pytest.mark.parametrize(
    "query,expected",
    [
        ("GPT-6-A12", [0]),
        ('"Ada Lovelace"', [0]),
        ('"Acme Robotics"', [0, 2]),
        ('"Nova Phone X"', [0]),
        ("$NVDA", [0, 2]),
        ("BRK.B", [0]),
        ("gpt-6-a12", [0]),
        ("GPT-6-B12", [1]),
        ("NonexistentProduct", []),
    ],
)
def test_names_tickers_identifiers_products_and_phrases(lexical_setup, query, expected):
    _, service, records = lexical_setup
    result = search(service, query)
    assert {hit.article_id for hit in result.results} == {records[i].id for i in expected}
    assert result.method == "lexical" and result.collection is None and result.model_version is None
    assert all(0 <= hit.relevance_score < 1 for hit in result.results)


def test_boolean_query_and_negative_only_query(lexical_setup):
    _, service, records = lexical_setup
    assert {hit.article_id for hit in search(service, '"Ada Lovelace" OR Byron').results} == {
        records[0].id,
        records[1].id,
    }
    assert [hit.article_id for hit in search(service, "Ada -Byron -Lovelace").results] == []
    assert search(service, "-Byron").count == 0
    assert search(service, '"" OR').count == 0


def test_phrase_cannot_cross_field_boundary(lexical_setup):
    _, service, records = lexical_setup
    assert records[3].id not in {r.article_id for r in search(service, '"Ada Lovelace"').results}
    assert records[3].id in {r.article_id for r in search(service, "Ada Lovelace").results}


@pytest.mark.parametrize(
    "filters,expected",
    [
        ({"source": "Other"}, [2]),
        ({"category": "Business"}, [2]),
        ({"language": "unknown"}, [2]),
        ({"language": "en"}, [0]),
        ({"date_from": "2026-09-09", "date_to": "2026-09-09"}, [0]),
        ({"date_from": "2026-09-10"}, []),
        ({"source": "Other", "category": "Business", "language": "unknown"}, [2]),
        ({"source": "' OR 1=1 --"}, []),
    ],
)
def test_sql_filters_apply_to_keyword_results(lexical_setup, filters, expected):
    _, service, records = lexical_setup
    assert {r.article_id for r in search(service, "NVDA", **filters).results} == {
        records[i].id for i in expected
    }


def test_direct_updates_are_searchable_and_rollback_restores_index(lexical_setup):
    sessions, service, records = lexical_setup
    with sessions.begin() as session:
        session.get(ArticleRecord, records[1].id).content = "FreshProduct Z99 released."
    assert search(service, '"FreshProduct Z99"').results[0].article_id == records[1].id
    with sessions() as session:
        session.get(ArticleRecord, records[1].id).content = "RollbackProduct"
        session.flush()
        session.rollback()
    assert search(service, "RollbackProduct").count == 0
    assert search(service, "FreshProduct").count == 1


def test_rejected_legacy_and_deleted_articles_are_excluded(lexical_setup):
    sessions, service, records = lexical_setup
    with sessions.begin() as session:
        session.get(ArticleRecord, records[0].id).processing_status = "rejected"
        session.get(ArticleRecord, records[2].id).cleaning_version = None
        session.delete(session.get(ArticleRecord, records[1].id))
    assert search(service, "NVDA OR Byron OR GPT").count == 0


def test_rank_limit_and_stable_tie_breaking(lexical_setup):
    sessions, service, _ = lexical_setup
    with sessions.begin() as session:
        a = session.scalar(select(ArticleRecord).order_by(ArticleRecord.id))
        b = ArticleRecord(
            title=a.title,
            description=a.description,
            content=a.content,
            source=a.source,
            category=a.category,
            retrieved_at=a.retrieved_at,
            url="https://example.com/copy",
            cleaning_version="1",
            processing_status="clean",
        )
        session.add(b)
    result = search(service, "GPT-6-A12", limit=1)
    assert result.count == 1 and result.results[0].article_id == a.id


def test_expression_index_exists_and_can_be_used(lexical_setup, database):
    engine = database[0]
    assert "ix_articles_lexical" in {i["name"] for i in inspect(engine).get_indexes("articles")}
    with engine.begin() as connection:
        connection.execute(text("SET LOCAL enable_seqscan = off"))
        plan = (
            connection.execute(
                text("""
            EXPLAIN SELECT id FROM articles WHERE
            newslens_lexical_vector(title, description, content)
            @@ websearch_to_tsquery('pg_catalog.simple', :query)
        """),
                {"query": "GPT-6-A12"},
            )
            .scalars()
            .all()
        )
    assert "ix_articles_lexical" in " ".join(plan)


def test_migration_indexes_existing_rows_and_downgrades_cleanly(lexical_setup, database):
    _, service, _ = lexical_setup
    engine, config, _ = database
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "0010")
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 4
        connection.execute(
            text(
                "UPDATE articles SET title='MigrationProduct' "
                "WHERE id=(SELECT min(id) FROM articles)"
            )
        )
        command.upgrade(config, "head")
        command.check(config)
    assert search(service, "MigrationProduct").count == 1


def test_runtime_and_api_need_no_embedding_model_or_qdrant(database, lexical_setup, monkeypatch):
    _, _, settings = database
    _, _, records = lexical_setup
    monkeypatch.setattr("backend.app.search_runtime.Settings", lambda: settings)

    def forbidden(*args, **kwargs):
        pytest.fail("Lexical search must not open Qdrant or load an embedding model")

    monkeypatch.setattr("backend.app.search_runtime.provider_for", forbidden)
    monkeypatch.setattr("backend.app.search_runtime.open_store", forbidden)
    with TestClient(create_app()) as client:
        response = client.get("/search", params={"q": "GPT-6-A12", "method": "lexical"})
    assert response.status_code == 200
    assert response.json()["results"][0]["article_id"] == records[0].id
    assert response.json()["method"] == "lexical"


def test_boundaries_preserve_ingestion_of_long_fields(lexical_setup):
    sessions, service, records = lexical_setup
    with sessions.begin() as session:
        session.get(ArticleRecord, records[0].id).content = (
            "PrefixProduct " + "a " * 25000 + "TailProduct"
        )
    assert search(service, "PrefixProduct").count == 1
    assert search(service, "TailProduct").count == 0
