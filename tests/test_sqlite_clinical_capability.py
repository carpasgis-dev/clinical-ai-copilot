"""Tests ``SqliteClinicalCapability`` (SQLite solo lectura)."""
from __future__ import annotations

import sqlite3

from app.capabilities.clinical_sql.sqlite_clinical_capability import SqliteClinicalCapability
from app.capabilities.contracts import ClinicalCapability


def test_sqlite_list_tables_and_select(tmp_path) -> None:
    db = tmp_path / "demo.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO patients (name) VALUES ('A')")
    conn.commit()
    conn.close()

    cap = SqliteClinicalCapability(db_path=str(db))
    assert isinstance(cap, ClinicalCapability)
    assert "patients" in cap.list_tables()
    assert cap.health_check() is True

    r = cap.run_safe_query("SELECT id, name FROM patients")
    assert r.error is None
    assert r.row_count == 1
    assert r.rows[0]["name"] == "A"


def test_sqlite_get_table_columns(tmp_path) -> None:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (id TEXT, birthdate TEXT)")
    conn.commit()
    conn.close()
    cap = SqliteClinicalCapability(db_path=str(db))
    cols = cap.get_table_columns("patients")
    assert "id" in cols and "birthdate" in cols


def test_sqlite_rejects_unknown_table(tmp_path) -> None:
    db = tmp_path / "demo.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    cap = SqliteClinicalCapability(db_path=str(db))
    r = cap.run_safe_query("SELECT id FROM secret_table")
    assert r.error
    assert "no permitida" in r.error.lower()


def test_sqlite_rejects_sqlite_master(tmp_path) -> None:
    db = tmp_path / "demo.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    cap = SqliteClinicalCapability(db_path=str(db))
    r = cap.run_safe_query("SELECT name FROM sqlite_master")
    assert r.error
    assert "sistema" in r.error.lower() or "no permitida" in r.error.lower()


def test_sqlite_honors_limit_with_parens(tmp_path) -> None:
    db = tmp_path / "demo.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT)")
    for i in range(5):
        conn.execute("INSERT INTO patients (name) VALUES (?)", (f"P{i}",))
    conn.commit()
    conn.close()
    cap = SqliteClinicalCapability(db_path=str(db))
    r = cap.run_safe_query("SELECT name FROM patients LIMIT (3)")
    assert r.error is None
    assert r.row_count == 3


def test_sqlite_rejects_non_literal_limit(tmp_path) -> None:
    db = tmp_path / "demo.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    cap = SqliteClinicalCapability(db_path=str(db))
    r = cap.run_safe_query("SELECT id FROM patients LIMIT (SELECT 1)")
    assert r.error
    assert "limit" in r.error.lower()


def test_sqlite_rejects_write(tmp_path) -> None:
    db = tmp_path / "x.db"
    c = sqlite3.connect(str(db))
    c.execute("CREATE TABLE t (x INT)")
    c.commit()
    c.close()
    cap = SqliteClinicalCapability(db_path=str(db))
    r = cap.run_safe_query("DELETE FROM t")
    assert r.error


def test_extract_summary_empty() -> None:
    cap = SqliteClinicalCapability(db_path="")
    assert cap.extract_clinical_summary("diabetes").conditions == []


def test_extract_summary_synthea_like(tmp_path) -> None:
    db = tmp_path / "syn.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE patients (id TEXT, birthdate TEXT, deathdate TEXT)")
    conn.execute(
        "INSERT INTO patients VALUES ('1', '1950-01-01', ''), ('2', '1940-06-01', '2020-01-01')"
    )
    conn.execute("CREATE TABLE conditions (patient TEXT, description TEXT)")
    conn.execute("INSERT INTO conditions VALUES ('1', 'Diabetes'), ('1', 'Hypertension')")
    conn.execute("CREATE TABLE medications (patient TEXT, description TEXT)")
    conn.execute("INSERT INTO medications VALUES ('1', 'Metformin')")
    conn.commit()
    conn.close()

    cap = SqliteClinicalCapability(db_path=str(db))
    ctx = cap.extract_clinical_summary("cualquier cosa")
    assert ctx.population_size == 1
    assert len(ctx.conditions) == 2
    assert "metformin" in [m.lower() for m in ctx.medications]
