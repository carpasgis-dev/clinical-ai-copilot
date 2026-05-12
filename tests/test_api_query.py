"""Tests de la API FastAPI (``POST /query``)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.graph_cache import clear_graph_cache


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_EVIDENCE_BACKEND", "stub")
    logf = tmp_path / "eval_runs.jsonl"
    monkeypatch.setenv("COPILOT_EVAL_LOG_PATH", str(logf))
    clear_graph_cache()
    from app.main import app

    with TestClient(app) as client:
        yield client, logf
    clear_graph_cache()


def test_post_query_smoke(api_client) -> None:
    client, logf = api_client
    r = client.post(
        "/query",
        json={"query": "¿Qué evidencia existe sobre metformina?", "session_id": "api-test-1"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["route"] == "evidence"
    assert data.get("sql_result") is None
    assert data["session_id"] == "api-test-1"
    assert "final_answer" in data
    assert data["trace"]
    assert data["pmids"]
    assert data["citations"]
    assert data["latency_ms"] >= 0
    assert data["ok"] is True
    assert data["citations"][0]["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/")
    lines = logf.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["ok"] is True
    assert row["query"]
    assert row["pmids"]


def test_post_query_hero_shape(api_client) -> None:
    client, _ = api_client
    hero = (
        "Paciente con diabetes e hipertensión mayor de 65 años. "
        "¿Qué evidencia reciente existe sobre tratamientos que "
        "reduzcan riesgo cardiovascular?"
    )
    r = client.post("/query", json={"query": hero})
    assert r.status_code == 200
    data = r.json()
    assert data["route"] == "hybrid"
    assert data["pubmed_query"]
    assert len(data["citations"]) >= 1


def test_health(api_client) -> None:
    client, _ = api_client
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "copilot_query_planner" in data
    assert "copilot_evidence_backend" in data
    assert data.get("clinical_db_loaded") in ("true", "false")
    assert "clinical_db_path" in data
    assert "copilot_llm_profile" in data


def test_root(api_client) -> None:
    client, _ = api_client
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["docs"] == "/docs"
    assert "query" in data


def test_post_query_sql_includes_executed_query(api_client) -> None:
    """Ruta SQL: la respuesta HTTP debe incluir el SELECT ejecutado (auditoría)."""
    client, _ = api_client
    r = client.post(
        "/query",
        json={
            "query": "cuantos pacientes con diabetes tenemos en nuestra base",
            "session_id": "api-sql-audit",
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["route"] == "sql"
    sr = data.get("sql_result")
    assert sr is not None, data
    assert "executed_query" in sr
    assert sr["executed_query"].strip().upper().startswith("SELECT"), sr["executed_query"]
    assert "row_count" in sr
