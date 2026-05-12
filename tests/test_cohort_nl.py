"""Tests heurística NL → SQL de cohorte (Synthea ETL)."""
from __future__ import annotations

from app.capabilities.clinical_sql.cohort_nl import (
    CohortNLSpec,
    build_synthea_cohort_count_sql,
    extract_cohort_nl_heuristic,
)


def test_extract_diabetes_metformin_age() -> None:
    q = (
        "¿Cuántos pacientes diabéticos mayores de 65 toman metformina "
        "en nuestra base de datos?"
    )
    spec = extract_cohort_nl_heuristic(q)
    assert "diabet" in spec.condition_like_tokens
    assert "metform" in spec.medication_like_tokens
    assert spec.min_age_years == 65


def test_build_count_sql_shape() -> None:
    spec = CohortNLSpec(
        condition_like_tokens=("diabet",),
        medication_like_tokens=("metform",),
        min_age_years=65,
    )
    sql, warns = build_synthea_cohort_count_sql(
        patient_cols={"id", "birthdate", "deathdate"},
        condition_cols={"patient", "description"},
        medication_cols={"patient", "description"},
        spec=spec,
    )
    assert sql
    assert "COUNT(*)" in sql and "cohort_size" in sql
    assert "conditions" in sql and "medications" in sql
    assert "65" in sql
    assert not any("error" in w.lower() for w in warns)


def test_build_fails_without_conditions_table() -> None:
    spec = CohortNLSpec(condition_like_tokens=("diabet",))
    sql, warns = build_synthea_cohort_count_sql(
        patient_cols={"id", "deathdate"},
        condition_cols=None,
        medication_cols=None,
        spec=spec,
    )
    assert sql is None
    assert warns
