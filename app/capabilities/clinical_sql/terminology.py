"""
Vocabulario de cohorte desde SQLite (Synthea / ETL).

Extrae términos reutilizables de ``conditions.description`` y ``medications.description``
para matching contra la consulta (motor sobre terminología, no lista cerrada enorme).
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from functools import lru_cache
from pathlib import Path


def fold_ascii(s: str) -> str:
    """Minúsculas y sin marcas diacríticas (índice de matching estable)."""
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# Palabras muy frecuentes en descripciones de medicación / ruido (no discriminantes).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "oral",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "patch",
        "spray",
        "solution",
        "suspension",
        "injection",
        "injectable",
        "unknown",
        "none",
        "prescription",
        "drug",
        "medication",
        "medications",
        "unknown",
        "mg",
        "ml",
        "mcg",
        "hr",
        "day",
        "daily",
        "once",
        "twice",
        "three",
        "times",
        "every",
        "hours",
        "as",
        "needed",
        "acute",
        "chronic",
        "history",
        "past",
        "active",
        "inactive",
        "primary",
        "secondary",
        "encounter",
        "snomed",
        "code",
    }
)

# Longitud mínima de token extraído (evita ruido tipo "oral").
_MIN_TOKEN_LEN = 5

# Límite de filas DISTINCT leídas (protección memoria / tiempo).
_DEFAULT_DISTINCT_LIMIT = 80_000


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)=lower(?) LIMIT 1",
        (table,),
    )
    return cur.fetchone() is not None


def _terms_from_description(desc: str) -> set[str]:
    """Palabras ``fold_ascii`` alfanuméricas ≥ ``_MIN_TOKEN_LEN`` fuera de stoplist."""
    folded = fold_ascii(desc.strip())
    out: set[str] = set()
    for m in re.finditer(r"[a-z0-9]{%d,}" % _MIN_TOKEN_LEN, folded):
        w = m.group()
        if w in _STOPWORDS:
            continue
        out.add(w)
    return out


def load_known_conditions(
    conn: sqlite3.Connection,
    *,
    distinct_limit: int = _DEFAULT_DISTINCT_LIMIT,
) -> set[str]:
    """
    Conjunto de términos (folded) derivados de ``SELECT DISTINCT description`` en ``conditions``.
    Si la tabla no existe, devuelve conjunto vacío.
    """
    if not _table_exists(conn, "conditions"):
        return set()
    cur = conn.execute(
        "SELECT DISTINCT TRIM(description) AS d FROM conditions "
        "WHERE TRIM(COALESCE(description, '')) != '' LIMIT ?",
        (int(distinct_limit),),
    )
    terms: set[str] = set()
    for (d,) in cur.fetchall():
        if not d:
            continue
        terms.update(_terms_from_description(str(d)))
    return terms


def load_known_medications(
    conn: sqlite3.Connection,
    *,
    distinct_limit: int = _DEFAULT_DISTINCT_LIMIT,
) -> set[str]:
    """Igual que ``load_known_conditions`` para la tabla ``medications``."""
    if not _table_exists(conn, "medications"):
        return set()
    cur = conn.execute(
        "SELECT DISTINCT TRIM(description) AS d FROM medications "
        "WHERE TRIM(COALESCE(description, '')) != '' LIMIT ?",
        (int(distinct_limit),),
    )
    terms: set[str] = set()
    for (d,) in cur.fetchall():
        if not d:
            continue
        terms.update(_terms_from_description(str(d)))
    return terms


@lru_cache(maxsize=32)
def _cached_condition_terms(cache_key: tuple[str, float]) -> frozenset[str]:
    path_str, _mtime = cache_key
    conn = sqlite3.connect(path_str, timeout=10.0)
    try:
        return frozenset(load_known_conditions(conn))
    finally:
        conn.close()


@lru_cache(maxsize=32)
def _cached_medication_terms(cache_key: tuple[str, float]) -> frozenset[str]:
    path_str, _mtime = cache_key
    conn = sqlite3.connect(path_str, timeout=10.0)
    try:
        return frozenset(load_known_medications(conn))
    finally:
        conn.close()


def terminology_cache_key(
    db_path: str,
    *,
    repo_root: Path | None = None,
) -> tuple[str, float]:
    """Clave (ruta resuelta, mtime) para invalidar caché al cambiar el fichero."""
    raw = (db_path or "").strip()
    if not raw:
        return ("", 0.0)
    path = Path(raw).expanduser()
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        return ("", 0.0)
    return str(path), path.stat().st_mtime


def load_cached_terminology(db_path: str) -> tuple[frozenset[str], frozenset[str]]:
    """
    Carga vocabulario condición/medicación con caché por ruta absoluta+mtime.

    Rutas relativas se resuelven desde la raíz del repo (misma convención que ``CLINICAL_DB_PATH``).
    """
    root = Path(__file__).resolve().parents[3]
    key = terminology_cache_key(db_path, repo_root=root)
    if not key[0]:
        return frozenset(), frozenset()
    return _cached_condition_terms(key), _cached_medication_terms(key)


def clear_terminology_cache() -> None:
    """Tests o recarga forzada tras sustituir el fichero SQLite."""
    _cached_condition_terms.cache_clear()
    _cached_medication_terms.cache_clear()
