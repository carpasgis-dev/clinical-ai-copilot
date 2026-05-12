"""
SQLite de solo lectura para ``ClinicalCapability``.

Ruta: ``CLINICAL_DB_PATH`` (o constructor). Pensado para BD derivada del ETL Synthea
(``scripts/synthea_csv_to_sqlite.py``); ``extract_clinical_summary`` agrega ``patients`` /
``conditions`` / ``medications`` cuando existen.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

from app.schemas.copilot_state import ClinicalContext, SqlResult, _CLINICAL_MAX_LIST, _SQL_MAX_ROWS


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|pragma|replace|truncate)\b",
    re.I,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


class SqliteClinicalCapability:
    """Introspección y SELECT acotado sobre un fichero SQLite."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._path = (db_path or os.getenv("CLINICAL_DB_PATH", "") or "").strip()

    def _connect(self) -> sqlite3.Connection:
        if not self._path:
            raise FileNotFoundError("CLINICAL_DB_PATH vacío")
        path = Path(self._path).expanduser()
        if not path.is_absolute():
            path = (_REPO_ROOT / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        conn = sqlite3.connect(str(path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def list_tables(self) -> list[str]:
        if not self._path:
            return []
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND NOT (name GLOB 'sqlite*') ORDER BY name"
                )
                return [str(r[0]) for r in cur.fetchall()]
        except Exception:
            return []

    def get_table_columns(self, table: str) -> list[str]:
        """
        Nombres de columnas vía ``PRAGMA`` (no pasa por ``run_safe_query``, donde ``PRAGMA`` está vetado).
        """
        name = (table or "").strip()
        if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            return []
        try:
            with self._connect() as conn:
                cur = conn.execute(f"PRAGMA table_info({name})")
                return [str(r[1]) for r in cur.fetchall()]
        except Exception:
            return []

    def run_safe_query(self, sql: str) -> SqlResult:
        raw = (sql or "").strip()
        if not raw:
            return SqlResult(executed_query="", error="consulta vacía")
        if ";" in raw.rstrip().rstrip(";"):
            return SqlResult(executed_query=raw, error="una sola sentencia; no uses ';' encadenado")
        low = raw.lower()
        if not low.startswith("select") and not low.startswith("with"):
            return SqlResult(executed_query=raw, error="solo consultas SELECT / WITH … SELECT")
        if _FORBIDDEN.search(raw):
            return SqlResult(executed_query=raw, error="palabra clave no permitida en SQL de lectura")
        bounded = raw.rstrip().rstrip(";")
        if not re.search(r"\blimit\s+\d+\s*$", low):
            bounded = f"{bounded} LIMIT {_SQL_MAX_ROWS}"

        try:
            with self._connect() as conn:
                cur = conn.execute(bounded)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows: list[dict[str, Any]] = []
                for row in cur:
                    rows.append({cols[i]: row[i] for i in range(len(cols))})
        except Exception as exc:
            return SqlResult(executed_query=bounded, error=str(exc))

        return SqlResult(
            executed_query=bounded,
            rows=rows,
            row_count=len(rows),
            tables_used=[],
        )

    def extract_clinical_summary(self, free_text_query: str) -> ClinicalContext:
        """
        Agrega cohorte desde tablas típicas del ETL Synthea (``patients``, ``conditions``,
        ``medications``). No vuelca filas: solo conteos y listas acotadas.
        """
        _ = free_text_query
        tables = {t.lower() for t in self.list_tables()}
        if "patients" not in tables:
            return ClinicalContext()

        def _cols(conn: sqlite3.Connection, t: str) -> set[str]:
            cur = conn.execute(f"PRAGMA table_info({t})")
            return {str(r[1]).lower() for r in cur.fetchall()}

        try:
            with self._connect() as conn:
                pcols = _cols(conn, "patients")
                alive_sql = "SELECT COUNT(*) AS n FROM patients"
                if "deathdate" in pcols:
                    alive_sql += " WHERE COALESCE(TRIM(deathdate), '') = ''"
                n_alive = int(conn.execute(alive_sql).fetchone()[0])

                age_range: str | None = None
                if "birthdate" in pcols and "deathdate" in pcols:
                    row = conn.execute(
                        "SELECT "
                        "MIN(CAST((julianday('now') - julianday(birthdate)) / 365.25 AS INT)), "
                        "MAX(CAST((julianday('now') - julianday(birthdate)) / 365.25 AS INT)) "
                        "FROM patients "
                        "WHERE COALESCE(TRIM(deathdate), '') = '' "
                        "AND birthdate IS NOT NULL AND TRIM(birthdate) != '' "
                        "AND julianday(birthdate) IS NOT NULL"
                    ).fetchone()
                    if row and row[0] is not None and row[1] is not None:
                        lo, hi = int(row[0]), int(row[1])
                        age_range = f"{lo}-{hi}" if lo != hi else str(lo)

                conditions: list[str] = []
                if "conditions" in tables:
                    ccols = _cols(conn, "conditions")
                    if "description" in ccols:
                        cur = conn.execute(
                            "SELECT DISTINCT TRIM(description) AS d FROM conditions "
                            "WHERE TRIM(COALESCE(description, '')) != '' "
                            "ORDER BY d LIMIT ?",
                            (_CLINICAL_MAX_LIST,),
                        )
                        conditions = [str(r[0]) for r in cur.fetchall() if r[0]]

                medications: list[str] = []
                if "medications" in tables:
                    mcols = _cols(conn, "medications")
                    if "description" in mcols:
                        cur = conn.execute(
                            "SELECT DISTINCT TRIM(description) AS d FROM medications "
                            "WHERE TRIM(COALESCE(description, '')) != '' "
                            "ORDER BY d LIMIT ?",
                            (_CLINICAL_MAX_LIST,),
                        )
                        medications = [str(r[0]) for r in cur.fetchall() if r[0]]

            return ClinicalContext(
                age_range=age_range,
                conditions=conditions,
                medications=medications,
                population_size=n_alive,
                population_hint="Cohorte importada (Synthea → SQLite)",
            )
        except Exception:
            return ClinicalContext()

    def health_check(self) -> bool:
        return bool(self._path) and bool(self.list_tables())
