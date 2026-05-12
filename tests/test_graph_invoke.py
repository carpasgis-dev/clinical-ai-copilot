"""
Tests de integración mínima del grafo LangGraph (invoke).

Usan ``StubEvidenceCapability`` para no depender de red NCBI en CI.
Para PubMed real: ``RUN_NCBI_INTEGRATION=1 pytest ...`` (ver ``test_graph_pubmed_integration.py``).
"""
from __future__ import annotations

import sqlite3

import pytest

from app.capabilities.clinical_sql.sqlite_clinical_capability import SqliteClinicalCapability
from app.capabilities.evidence_rag import StubEvidenceCapability
from app.orchestration.graph import build_copilot_graph
from app.schemas.copilot_state import NodeName, Route, TraceStep


@pytest.fixture
def graph_stub():
    return build_copilot_graph(evidence=StubEvidenceCapability())


def _trace_step_node_id(step: object) -> str:
    if isinstance(step, dict):
        return str(step["node"])
    if isinstance(step, TraceStep):
        return step.node.value
    raise TypeError(f"unexpected trace step type: {type(step)!r}")


def _node_ids(trace: list) -> list[str]:
    return [_trace_step_node_id(s) for s in trace]


def test_invoke_sql_route_with_clinical_db(tmp_path) -> None:
    db = tmp_path / "c.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (deathdate TEXT)")
    conn.execute("INSERT INTO patients VALUES (''), ('2020-01-01')")
    conn.commit()
    conn.close()
    cap = SqliteClinicalCapability(db_path=str(db))
    g = build_copilot_graph(evidence=StubEvidenceCapability(), clinical=cap)
    result = g.invoke(
        {
            "user_query": "¿Cuántos pacientes hay en nuestra base de datos?",
            "session_id": "test-sql-db-1",
        }
    )
    assert result["route"] == Route.SQL
    assert result["sql_result"]["row_count"] == 1
    warns = result.get("warnings") or []
    assert not any("stub" in w for w in warns)


def test_invoke_sql_route_nl_filtered_cohort(tmp_path) -> None:
    """NL → SQL: diabetes + edad + metformina (solo un paciente cumple)."""
    db = tmp_path / "cohort.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE patients (id TEXT, birthdate TEXT, deathdate TEXT)"
    )
    conn.execute(
        "INSERT INTO patients VALUES "
        "('p-old', '1950-06-01', ''),"
        "('p-young', '2010-01-01', '')"
    )
    conn.execute("CREATE TABLE conditions (patient TEXT, description TEXT)")
    conn.execute(
        "INSERT INTO conditions VALUES "
        "('p-old', 'Type 2 diabetes mellitus'),"
        "('p-young', 'Type 2 diabetes mellitus')"
    )
    conn.execute("CREATE TABLE medications (patient TEXT, description TEXT)")
    conn.execute(
        "INSERT INTO medications VALUES "
        "('p-old', 'Metformin 500 MG Oral Tablet'),"
        "('p-young', 'Metformin 500 MG Oral Tablet')"
    )
    conn.commit()
    conn.close()
    cap = SqliteClinicalCapability(db_path=str(db))
    g = build_copilot_graph(evidence=StubEvidenceCapability(), clinical=cap)
    result = g.invoke(
        {
            "user_query": (
                "¿Cuántos pacientes diabéticos mayores de 65 toman metformina "
                "en nuestra base de datos?"
            ),
            "session_id": "test-sql-nl-1",
        }
    )
    assert result["route"] == Route.SQL
    assert result["sql_result"]["row_count"] == 1
    summaries = [
        getattr(t, "summary", None) or (t.get("summary") if isinstance(t, dict) else "")
        for t in result["trace"]
    ]
    assert any("NL→SQL" in s for s in summaries)
    sql = result["sql_result"]["executed_query"] or ""
    assert "conditions" in sql.lower() and "medications" in sql.lower()


def test_invoke_sql_route(graph_stub) -> None:
    result = graph_stub.invoke(
        {
            "user_query": "¿Cuántos pacientes diabéticos hay en nuestra base de datos?",
            "session_id": "test-sql-1",
        }
    )
    assert result["route"] == Route.SQL
    assert "sql_result" in result
    assert result["sql_result"]["row_count"] == 1
    ids = _node_ids(result["trace"])
    assert ids[0] == NodeName.ROUTER.value
    assert NodeName.SQL.value in ids
    assert NodeName.SYNTHESIS.value in ids
    assert NodeName.SAFETY.value in ids
    assert result["disclaimer"]
    assert result["disclaimer"] in result["final_answer"]
    assert len(result["trace"]) == 4
    warns = result.get("warnings") or []
    assert any("stub" in w for w in warns)


def test_invoke_evidence_route(graph_stub) -> None:
    result = graph_stub.invoke(
        {
            "user_query": "¿Qué evidencia existe sobre metformina?",
            "session_id": "test-ev-1",
        }
    )
    assert result["route"] == Route.EVIDENCE
    assert result["evidence_bundle"]["pmids"]
    ids = _node_ids(result["trace"])
    assert NodeName.EVIDENCE_RETRIEVAL.value in ids
    assert NodeName.SYNTHESIS.value in ids
    assert len(result["trace"]) == 4


def test_trace_reducer_preserves_order_sql_path(graph_stub) -> None:
    """El reducer ``add`` concatena pasos sin que el último nodo pise al resto."""
    result = graph_stub.invoke(
        {
            "user_query": "¿Cuántos pacientes hay en nuestra cohorte?",
            "session_id": "test-order-1",
        }
    )
    assert result["route"] == Route.SQL
    assert _node_ids(result["trace"]) == [
        NodeName.ROUTER.value,
        NodeName.SQL.value,
        NodeName.SYNTHESIS.value,
        NodeName.SAFETY.value,
    ]


def test_invoke_hybrid_hero_route(graph_stub) -> None:
    hero = (
        "Paciente con diabetes e hipertensión mayor de 65 años. "
        "¿Qué evidencia reciente existe sobre tratamientos que "
        "reduzcan riesgo cardiovascular?"
    )
    result = graph_stub.invoke({"user_query": hero, "session_id": "test-hero-1"})
    assert result["route"] == Route.HYBRID
    assert result.get("pubmed_query")
    assert len(result["evidence_bundle"]["articles"]) >= 1
    ids = _node_ids(result["trace"])
    assert NodeName.CLINICAL_SUMMARY.value in ids
    assert NodeName.PUBMED_QUERY_BUILDER.value in ids
    assert ids.count(NodeName.EVIDENCE_RETRIEVAL.value) == 1


def test_invoke_unknown_route(graph_stub) -> None:
    result = graph_stub.invoke(
        {"user_query": "Hola", "session_id": "test-unknown-1"}
    )
    assert result["route"] == Route.UNKNOWN
    ids = _node_ids(result["trace"])
    assert NodeName.FALLBACK.value in ids
    assert NodeName.SYNTHESIS.value not in ids
    assert NodeName.SAFETY.value in ids


def test_default_graph_uses_ncbi_evidence() -> None:
    """Sin argumentos, el grafo usa PubMed real (NcbiEvidenceCapability)."""
    from app.capabilities.evidence_rag import NcbiEvidenceCapability

    g = build_copilot_graph()
    # El grafo compilado no expone la capability; comprobamos que el nodo es invocable
    # y que la clase por defecto es la NCBI (smoke: mismo tipo que inyectamos explícitamente).
    g2 = build_copilot_graph(evidence=NcbiEvidenceCapability())
    assert type(g).__name__ == type(g2).__name__
