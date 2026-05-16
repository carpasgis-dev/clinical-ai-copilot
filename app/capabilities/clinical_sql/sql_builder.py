"""
``CohortQuery`` → SQL ``SELECT COUNT(*) AS cohort_size`` (solo lectura, generado por código).
"""
from __future__ import annotations

from app.capabilities.clinical_sql.cohort_nl import CohortNLSpec, build_synthea_cohort_count_sql
from app.capabilities.clinical_sql.cohort_parser import CohortQuery


def cohort_query_to_nl_spec(q: CohortQuery) -> CohortNLSpec:
    sex = (q.sex or "").strip().upper()[:1]
    sex_norm = sex if sex in ("F", "M") else None
    return CohortNLSpec(
        condition_like_tokens=tuple(q.condition_like_tokens),
        medication_like_tokens=tuple(q.medication_like_tokens),
        min_age_years=q.age_min_years,
        max_age_years=q.age_max_years,
        alive_only=q.alive_only,
        sex=sex_norm,
    )


def build_sql_from_cohort(
    q: CohortQuery,
    *,
    patient_cols: set[str],
    condition_cols: set[str] | None,
    medication_cols: set[str] | None,
) -> tuple[str | None, list[str]]:
    spec = cohort_query_to_nl_spec(q)
    return build_synthea_cohort_count_sql(
        patient_cols=patient_cols,
        condition_cols=condition_cols,
        medication_cols=medication_cols,
        spec=spec,
    )
