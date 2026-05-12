"""
Heurísticas español/inglés → filtros de cohorte y SQL ``SELECT`` de solo conteo.

El SQL se **construye** solo a partir de tokens controlados (sin interpolar texto libre
del usuario). Pensado para el esquema del ETL Synthea (``patients``, ``conditions``,
``medications`` con columnas típicas ``id``, ``birthdate``, ``deathdate``, ``patient``,
``description``).

Inspiración de introspección / cohortes: proyecto hermano ``sina_mcp/sqlite-analyzer``
(LangChain/SQL allí es más abierto; aquí solo generación acotada + ``run_safe_query``).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# (subcadenas normalizadas que disparan el filtro, fragmento LIKE seguro [a-z0-9])
_CONDITION_TRIGGERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("diabet",), "diabet"),
    (("hipertens",), "hipertens"),
    (("asma",), "asma"),
    (("epoc", "copd",), "obstruct"),
    (("insuficiencia cardiaca", "heart failure", "insuf cardiaca"), "cardiac"),
)

_MED_TRIGGERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("metform",), "metform"),
    (("aspirin", "aspirina", "acido acetilsalicilico", "ácido acetilsalicílico"), "aspir"),
    (("insulin", "insulina",), "insulin"),
    (("losartan",), "losartan"),
    (("enalapril",), "enalapril"),
)


@dataclass
class CohortNLSpec:
    """Filtros inferidos de la pregunta (solo valores ya saneados para SQL)."""

    condition_like_tokens: tuple[str, ...] = ()
    medication_like_tokens: tuple[str, ...] = ()
    min_age_years: int | None = None
    max_age_years: int | None = None
    alive_only: bool = True

    def has_filters(self) -> bool:
        return bool(
            self.condition_like_tokens
            or self.medication_like_tokens
            or self.min_age_years is not None
            or self.max_age_years is not None
        )


def _fold(s: str) -> str:
    """Minúsculas y sin tildes para matching simple."""
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def extract_cohort_nl_heuristic(user_query: str) -> CohortNLSpec:
    q = _fold(user_query or "")

    conds: list[str] = []
    for triggers, like_tok in _CONDITION_TRIGGERS:
        if any(t in q for t in triggers):
            if like_tok not in conds:
                conds.append(like_tok)

    meds: list[str] = []
    for triggers, like_tok in _MED_TRIGGERS:
        if any(t in q for t in triggers):
            if like_tok not in meds:
                meds.append(like_tok)

    min_age: int | None = None
    max_age: int | None = None
    for pat in (
        r"mayores?\s+de\s+(\d{1,3})\b",
        r"mas\s+de\s+(\d{1,3})\b",
        r">\s*(\d{1,3})\b",
        r"desde\s+los?\s+(\d{1,3})\b",
    ):
        m = re.search(pat, q)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 120:
                min_age = v
            break
    for pat in (
        r"menores?\s+de\s+(\d{1,3})\b",
        r"menos\s+de\s+(\d{1,3})\b",
        r"<\s*(\d{1,3})\b",
    ):
        m = re.search(pat, q)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 120:
                max_age = v
            break
    m = re.search(r"\b(\d{1,3})\s*a(?:n|ñ)os?\b", q)
    if m and min_age is None and max_age is None:
        v = int(m.group(1))
        if 0 <= v <= 120:
            min_age = v

    return CohortNLSpec(
        condition_like_tokens=tuple(conds),
        medication_like_tokens=tuple(meds),
        min_age_years=min_age,
        max_age_years=max_age,
        alive_only=True,
    )


def _pick_col(cols: set[str], *candidates: str) -> str | None:
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def build_synthea_cohort_count_sql(
    *,
    patient_cols: set[str],
    condition_cols: set[str] | None,
    medication_cols: set[str] | None,
    spec: CohortNLSpec,
) -> tuple[str | None, list[str]]:
    """
    Devuelve ``(sql, warnings)``. ``sql`` es ``None`` si no se puede armar el filtro
    (faltan tablas o columnas) y el caller debe usar un conteo simple.
    """
    warns: list[str] = []
    pid = _pick_col(patient_cols, "id", "Id", "patient_id", "pat_id")
    birth = _pick_col(patient_cols, "birthdate", "date_of_birth", "dob")
    death = _pick_col(patient_cols, "deathdate", "death_date")

    if not pid:
        return None, ["sql: tabla patients sin columna id reconocible"]

    need_c = bool(spec.condition_like_tokens)
    need_m = bool(spec.medication_like_tokens)
    cp = cd = ""
    mp = md = ""

    if need_c:
        if not condition_cols:
            return None, ["sql: se pidió filtro por condición pero no hay tabla/columnas conditions"]
        cp = _pick_col(condition_cols, "patient", "patient_id", "pat_id", pid.lower()) or ""
        cd = _pick_col(condition_cols, "description", "desc", "code_text") or ""
        if not cp or not cd:
            return None, ["sql: conditions sin columnas patient/description"]

    if need_m:
        if not medication_cols:
            return None, ["sql: se pidió filtro por medicación pero no hay tabla/columnas medications"]
        mp = _pick_col(medication_cols, "patient", "patient_id", "pat_id", pid.lower()) or ""
        md = _pick_col(medication_cols, "description", "desc", "medication") or ""
        if not mp or not md:
            return None, ["sql: medications sin columnas patient/description"]

    age_parts: list[str] = []
    if spec.min_age_years is not None:
        if not birth:
            warns.append("sql: edad mínima ignorada (falta birthdate en patients)")
        else:
            lo = max(0, min(130, int(spec.min_age_years)))
            age_parts.append(
                f"(julianday('now') - julianday(p.\"{birth}\")) / 365.25 >= {lo}"
            )
    if spec.max_age_years is not None:
        if not birth:
            warns.append("sql: edad máxima ignorada (falta birthdate en patients)")
        else:
            hi = max(0, min(130, int(spec.max_age_years)))
            age_parts.append(
                f"(julianday('now') - julianday(p.\"{birth}\")) / 365.25 < {hi}"
            )

    where_alive = ""
    if spec.alive_only and death:
        where_alive = f' AND COALESCE(TRIM(p."{death}"), \'\') = \'\''

    where_age = ""
    if age_parts:
        where_age = " AND " + " AND ".join(age_parts)

    exists_c = ""
    for tok in spec.condition_like_tokens:
        safe = re.sub(r"[^a-z0-9]", "", tok.lower())[:40]
        if not safe:
            continue
        exists_c += (
            f' AND EXISTS (SELECT 1 FROM conditions c WHERE c."{cp}" = p."{pid}" '
            f'AND LOWER(COALESCE(c."{cd}", \'\')) LIKE \'%{safe}%\')'
        )

    exists_m = ""
    for tok in spec.medication_like_tokens:
        safe = re.sub(r"[^a-z0-9]", "", tok.lower())[:40]
        if not safe:
            continue
        exists_m += (
            f' AND EXISTS (SELECT 1 FROM medications m WHERE m."{mp}" = p."{pid}" '
            f'AND LOWER(COALESCE(m."{md}", \'\')) LIKE \'%{safe}%\')'
        )

    inner = (
        f'SELECT DISTINCT p."{pid}" AS _pid FROM patients p WHERE 1=1'
        f"{where_alive}{where_age}{exists_c}{exists_m}"
    )
    sql = f"SELECT COUNT(*) AS cohort_size FROM ({inner}) AS _cohort"
    return sql, warns
