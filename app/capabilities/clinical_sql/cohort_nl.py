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
from dataclasses import dataclass


@dataclass
class CohortNLSpec:
    """Filtros inferidos de la pregunta (solo valores ya saneados para SQL)."""

    condition_like_tokens: tuple[str, ...] = ()
    medication_like_tokens: tuple[str, ...] = ()
    min_age_years: int | None = None
    max_age_years: int | None = None
    alive_only: bool = True
    sex: str | None = None  # "F" | "M" — columna ``gender``/``sex`` en patients si existe

    def has_filters(self) -> bool:
        return bool(
            self.condition_like_tokens
            or self.medication_like_tokens
            or self.min_age_years is not None
            or self.max_age_years is not None
            or (self.sex is not None and str(self.sex).strip() != "")
        )


def extract_cohort_nl_heuristic(user_query: str) -> CohortNLSpec:
    """Delega en el parser estructurado (``clinical_sql``) para una sola fuente de heurísticas."""
    from app.capabilities.clinical_sql.cohort_parser import parse_cohort_query
    from app.capabilities.clinical_sql.sql_builder import cohort_query_to_nl_spec

    return cohort_query_to_nl_spec(parse_cohort_query(user_query))


def _pick_col(cols: set[str], *candidates: str) -> str | None:
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _group_like_tokens_for_or(tokens: tuple[str, ...]) -> list[tuple[str, ...]]:
    """
    Agrupa tokens que son variantes/sinónimos (substring o prefijo común) para un OR interno.

    Distintos grupos se combinan con AND (p. ej. diabetes + hipertensión).
    """
    uniq: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        safe = re.sub(r"[^a-z0-9]", "", tok.lower())[:40]
        if not safe or safe in seen:
            continue
        seen.add(safe)
        uniq.append(safe)
    if not uniq:
        return []

    def related(a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a in b or b in a:
            return True
        if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
            return True
        return False

    parent = list(range(len(uniq)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if related(uniq[i], uniq[j]):
                union(i, j)

    buckets: dict[int, list[str]] = {}
    order_roots: list[int] = []
    for i in range(len(uniq)):
        r = find(i)
        if r not in buckets:
            order_roots.append(r)
            buckets[r] = []
        if uniq[i] not in buckets[r]:
            buckets[r].append(uniq[i])
    return [tuple(buckets[r]) for r in order_roots]


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

    where_sex = ""
    sx = (spec.sex or "").strip().upper()[:1]
    if sx in ("F", "M"):
        gcol = _pick_col(patient_cols, "gender", "sex")
        if gcol:
            where_sex = f' AND UPPER(SUBSTR(TRIM(p."{gcol}"), 1, 1)) = \'{sx}\''
        else:
            warns.append("sql: filtro sexo ignorado (sin columna gender/sex en patients)")

    where_age = ""
    if age_parts:
        where_age = " AND " + " AND ".join(age_parts)

    exists_c = ""
    for group in _group_like_tokens_for_or(spec.condition_like_tokens):
        if not group:
            continue
        if len(group) == 1:
            safe = group[0]
            exists_c += (
                f' AND EXISTS (SELECT 1 FROM conditions c WHERE c."{cp}" = p."{pid}" '
                f'AND LOWER(COALESCE(c."{cd}", \'\')) LIKE \'%{safe}%\')'
            )
        else:
            ors = " OR ".join(
                f'LOWER(COALESCE(c."{cd}", \'\')) LIKE \'%{tok}%\''
                for tok in group
            )
            exists_c += (
                f' AND EXISTS (SELECT 1 FROM conditions c WHERE c."{cp}" = p."{pid}" '
                f"AND ({ors}))"
            )

    exists_m = ""
    for group in _group_like_tokens_for_or(spec.medication_like_tokens):
        if not group:
            continue
        if len(group) == 1:
            safe = group[0]
            exists_m += (
                f' AND EXISTS (SELECT 1 FROM medications m WHERE m."{mp}" = p."{pid}" '
                f'AND LOWER(COALESCE(m."{md}", \'\')) LIKE \'%{safe}%\')'
            )
        else:
            ors = " OR ".join(
                f'LOWER(COALESCE(m."{md}", \'\')) LIKE \'%{tok}%\''
                for tok in group
            )
            exists_m += (
                f' AND EXISTS (SELECT 1 FROM medications m WHERE m."{mp}" = p."{pid}" '
                f"AND ({ors}))"
            )

    inner = (
        f'SELECT DISTINCT p."{pid}" AS _pid FROM patients p WHERE 1=1'
        f"{where_alive}{where_sex}{where_age}{exists_c}{exists_m}"
    )
    sql = f"SELECT COUNT(*) AS cohort_size FROM ({inner}) AS _cohort"
    return sql, warns
