"""Fase 3.3 — clarify node y resolución vía sesión."""
from __future__ import annotations

import sqlite3

from app.capabilities.clinical_sql.sqlite_clinical_capability import SqliteClinicalCapability
from app.capabilities.evidence_rag import StubEvidenceCapability
from app.orchestration.graph import build_copilot_graph
from app.schemas.copilot_state import Route
from app.session.memory import load_session_memory


def test_clarify_two_turns_resolves_to_sql(tmp_path) -> None:
    """Tras ambiguous, la respuesta «solo datos…» fuerza SQL y limpia pending."""
    db = tmp_path / "clarify.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (id TEXT, birthdate TEXT, deathdate TEXT)")
    conn.execute("INSERT INTO patients VALUES ('p1', '1950-01-01', '')")
    conn.execute("CREATE TABLE conditions (patient TEXT, description TEXT)")
    conn.execute(
        "INSERT INTO conditions VALUES ('p1', 'Type 2 diabetes mellitus')"
    )
    conn.commit()
    conn.close()
    cap = SqliteClinicalCapability(db_path=str(db))
    g = build_copilot_graph(evidence=StubEvidenceCapability(), clinical=cap)
    sid = "clarify-session-1"
    r1 = g.invoke(
        {
            "user_query": "pacientes cohorte evidencia efectividad",
            "session_id": sid,
        }
    )
    assert r1["route"] == Route.AMBIGUOUS
    assert load_session_memory(sid).pending_clarification is True
    r2 = g.invoke(
        {
            "user_query": "solo datos de la cohorte en nuestra base",
            "session_id": sid,
        }
    )
    assert r2["route"] == Route.SQL
    assert load_session_memory(sid).pending_clarification is False
    assert "resolved" in (r2.get("route_reason") or "").lower() or "post_clarify" in (
        r2.get("route_reason") or ""
    )
