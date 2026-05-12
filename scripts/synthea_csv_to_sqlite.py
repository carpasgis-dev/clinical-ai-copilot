#!/usr/bin/env python3
"""
ETL mínimo: CSV exportados por Synthea → SQLite para ``SqliteClinicalCapability``.

Tablas: ``patients``, ``conditions``, ``medications`` (solo si existen los CSV).
Uso desde la raíz del repo ``clinical-ai-copilot``::

    python scripts/synthea_csv_to_sqlite.py
    python scripts/synthea_csv_to_sqlite.py --csv-dir data/synthea/output/csv --db data/clinical/synthea.db

Variables opcionales: ``SYNTHEA_CSV_DIR``, ``CLINICAL_DB_PATH``.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable


def _sanitize(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", (name or "").strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "col"


def _dedupe_columns(names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        base = _sanitize(n)
        k = base
        if k not in seen:
            seen[k] = 0
        else:
            seen[k] += 1
            k = f"{base}_{seen[base]}"
        out.append(k)
    return out


def _import_one(conn: sqlite3.Connection, csv_path: Path, table: str) -> int:
    if not csv_path.is_file():
        return 0
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return 0
        cols = _dedupe_columns(reader.fieldnames)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(
            "CREATE TABLE {} ({})".format(
                table,
                ", ".join(f'"{c}" TEXT' for c in cols),
            )
        )
        placeholders = ", ".join(["?"] * len(cols))
        col_sql = ", ".join(f'"{c}"' for c in cols)
        sql = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"
        n = 0
        for row in reader:
            values = []
            for old, new in zip(reader.fieldnames, cols):
                v = row.get(old)
                values.append("" if v is None else str(v))
            conn.execute(sql, values)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthea CSV → SQLite (tablas patients/conditions/medications)")
    ap.add_argument(
        "--csv-dir",
        default=os.getenv("SYNTHEA_CSV_DIR", "data/synthea/output/csv"),
        help="Carpeta con patients.csv, conditions.csv, medications.csv",
    )
    ap.add_argument(
        "--db",
        default=os.getenv("CLINICAL_DB_PATH", "data/clinical/synthea.db"),
        help="Fichero SQLite de salida",
    )
    args = ap.parse_args()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("patients.csv", "patients"),
        ("conditions.csv", "conditions"),
        ("medications.csv", "medications"),
    ]
    conn = sqlite3.connect(str(db_path))
    try:
        total = 0
        for fn, table in pairs:
            n = _import_one(conn, csv_dir / fn, table)
            print(f"{fn}: {n} filas → {table}")
            total += n
        conn.commit()
    finally:
        conn.close()
    print(f"SQLite escrito: {db_path} (total filas importadas: {total})")


if __name__ == "__main__":
    main()
