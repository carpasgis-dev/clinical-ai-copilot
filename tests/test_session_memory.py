"""Fase 3.2 — memoria de sesión y fusión de cohortes."""
from __future__ import annotations

import sqlite3

from app.capabilities.clinical_sql.cohort_parser import (
    CohortQuery,
    merge_cohort_queries,
    parse_cohort_query,
)
from app.capabilities.clinical_sql.sqlite_clinical_capability import SqliteClinicalCapability
from app.capabilities.clinical_sql.terminology import clear_terminology_cache
from app.capabilities.evidence_rag import StubEvidenceCapability
from app.orchestration.graph import build_copilot_graph
from app.orchestration.planner import build_execution_plan
from app.schemas.copilot_state import Route
from app.session.memory import (
    clear_session_memory_store,
    load_session_memory,
    update_session_after_planner,
    update_session_after_sql_route,
)


def test_merge_cohort_queries_unions_conditions_and_meds() -> None:
    old = CohortQuery(
        condition_like_tokens=["diabet"],
        medication_like_tokens=["metform"],
        age_min_years=65,
        sex=None,
        count_only=True,
        alive_only=True,
    )
    new = parse_cohort_query(
        "cuantos usan insulina",
        known_condition_terms=None,
        known_medication_terms=None,
    )
    m = merge_cohort_queries(old, new)
    assert "diabet" in m.condition_like_tokens
    assert "insulin" in m.medication_like_tokens
    assert "metform" in m.medication_like_tokens
    assert m.age_min_years == 65


def test_merge_sex_from_new_overrides() -> None:
    old = CohortQuery(
        condition_like_tokens=["diabet"],
        age_min_years=65,
        sex="M",
        count_only=True,
        alive_only=True,
    )
    new = parse_cohort_query(
        "solo mujeres diabéticas",
        known_condition_terms=None,
        known_medication_terms=None,
    )
    m = merge_cohort_queries(old, new)
    assert m.sex == "F"
    assert m.age_min_years == 65


def test_session_memory_store_roundtrip() -> None:
    clear_session_memory_store()
    cq = CohortQuery(condition_like_tokens=["asma"], count_only=True, alive_only=True)
    plan = build_execution_plan(Route.SQL)
    update_session_after_planner("s-1", Route.SQL, plan)
    update_session_after_sql_route(
        "s-1",
        route=Route.SQL,
        effective_cohort=cq,
        sql_executed="SELECT 1",
        structured_cohort_applied=True,
    )
    mem = load_session_memory("s-1")
    assert mem.last_route == Route.SQL
    assert mem.last_sql_executed == "SELECT 1"
    assert mem.last_cohort is not None
    assert mem.last_cohort.condition_like_tokens == ["asma"]


def test_graph_sql_two_turns_merge_follow_up(tmp_path) -> None:
    """Misma sesión: turno 2 añade insulina sobre diabetes+65+metformina del turno 1."""
    clear_terminology_cache()
    db = tmp_path / "sess.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (id TEXT, birthdate TEXT, deathdate TEXT)")
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
    sid = "session-merge-1"
    g.invoke(
        {
            "user_query": (
                "¿Cuántos pacientes diabéticos mayores de 65 toman metformina "
                "en nuestra base de datos?"
            ),
            "session_id": sid,
        }
    )
    r2 = g.invoke(
        {
            "user_query": "¿Y cuántos usan insulina?",
            "session_id": sid,
        }
    )
    assert r2["route"] == Route.SQL
    sql = (r2.get("sql_result") or {}).get("executed_query") or ""
    low = sql.lower()
    assert "insulin" in low or "insul" in low
    mem = load_session_memory(sid)
    assert mem.last_cohort is not None
    assert "insulin" in mem.last_cohort.medication_like_tokens
    assert "diabet" in mem.last_cohort.condition_like_tokens


def test_session_turn_snapshot_fields(tmp_path) -> None:
    """Tras invoke: ``last_query``, ``last_sql_result`` y ``last_medical_answer`` en RAM."""
    clear_terminology_cache()
    db = tmp_path / "snap2.db"
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
    sid = "snap-fields-1"
    g.invoke(
        {
            "user_query": "¿Cuántos pacientes diabéticos en nuestra base?",
            "session_id": sid,
        }
    )
    mem = load_session_memory(sid)
    assert mem.last_query
    assert mem.last_sql_result is not None
    assert "executed_query" in mem.last_sql_result
    assert mem.last_medical_answer is not None
