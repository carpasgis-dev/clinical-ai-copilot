"""Fase 3.5 — detección de follow-up y memoria conversacional."""
from __future__ import annotations

import sqlite3

from app.capabilities.clinical_sql.sqlite_clinical_capability import SqliteClinicalCapability
from app.capabilities.evidence_rag import StubEvidenceCapability
from app.capabilities.clinical_sql.terminology import clear_terminology_cache
from app.orchestration.graph import build_copilot_graph
from app.schemas.copilot_state import Route
from app.session.followup import is_followup_query
from app.session.memory import load_session_memory


def test_is_followup_query_prefixes() -> None:
    assert is_followup_query("¿y con insulina?")
    assert is_followup_query("y con metformina")
    assert is_followup_query("También solo mujeres")


def test_is_followup_query_new_topic_not_followup() -> None:
    assert not is_followup_query(
        "¿Cuántos pacientes hay con asma en nuestra base de datos?"
    )


def test_graph_saves_last_query_and_medical_answer(tmp_path) -> None:
    clear_terminology_cache()
    db = tmp_path / "snap.db"
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
    sid = "snap-session-1"
    q = "¿Cuántos pacientes diabéticos hay en nuestra base de datos?"
    g.invoke({"user_query": q, "session_id": sid})
    mem = load_session_memory(sid)
    assert mem.last_query == q
    assert mem.last_medical_answer is not None
    assert "summary" in mem.last_medical_answer


def test_non_followup_second_turn_does_not_merge_diabetes_cohort(tmp_path) -> None:
    """Consulta larga con señales de nueva cohorte: no fusiona con diabetes previa."""
    clear_terminology_cache()
    db = tmp_path / "split.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (id TEXT, birthdate TEXT, deathdate TEXT)")
    conn.execute(
        "INSERT INTO patients VALUES "
        "('p-dm', '1950-01-01', ''),"
        "('p-asma', '1955-01-01', '')"
    )
    conn.execute("CREATE TABLE conditions (patient TEXT, description TEXT)")
    conn.execute(
        "INSERT INTO conditions VALUES "
        "('p-dm', 'Type 2 diabetes mellitus'),"
        "('p-asma', 'Asthma')"
    )
    conn.commit()
    conn.close()
    cap = SqliteClinicalCapability(db_path=str(db))
    g = build_copilot_graph(evidence=StubEvidenceCapability(), clinical=cap)
    sid = "no-merge-1"
    g.invoke(
        {
            "user_query": (
                "¿Cuántos pacientes diabéticos mayores de 65 hay en nuestra base de datos?"
            ),
            "session_id": sid,
        }
    )
    r2 = g.invoke(
        {
            "user_query": "¿Cuántos pacientes hay con asma en nuestra base de datos?",
            "session_id": sid,
        }
    )
    assert r2["route"] == Route.SQL
    sql = (r2.get("sql_result") or {}).get("executed_query") or ""
    low = sql.lower()
    assert "asma" in low or "asthma" in low
    assert "diabet" not in low
