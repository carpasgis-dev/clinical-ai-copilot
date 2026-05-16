"""Tests parser estructurado ``CohortQuery`` + ``build_sql_from_cohort``."""
from __future__ import annotations

import sqlite3

from app.capabilities.clinical_sql.cohort_parser import (
    cohort_query_has_filters,
    parse_cohort_query,
)
from app.capabilities.clinical_sql.sql_builder import build_sql_from_cohort, cohort_query_to_nl_spec
from app.capabilities.clinical_sql.cohort_nl import extract_cohort_nl_heuristic
from app.capabilities.clinical_sql.terminology import (
    clear_terminology_cache,
    load_known_conditions,
    load_known_medications,
)


def test_parse_diabetes_metformin_age_sex() -> None:
    q = (
        "¿Cuántas mujeres diabéticas mayores de 65 toman metformina "
        "en nuestra base de datos?"
    )
    cq = parse_cohort_query(q)
    assert set(cq.condition_like_tokens) == {"diabet"}
    assert set(cq.medication_like_tokens) == {"metform"}
    assert cq.age_min_years == 65
    assert cq.sex == "F"
    assert cohort_query_has_filters(cq)


def test_parse_hombres_hipertension() -> None:
    cq = parse_cohort_query("cuántos hombres hipertensos vivos")
    assert "hypertens" in cq.condition_like_tokens
    assert cq.sex == "M"


def test_bridge_matches_legacy_extract() -> None:
    text = "pacientes con asma menores de 18"
    assert cohort_query_to_nl_spec(parse_cohort_query(text)) == extract_cohort_nl_heuristic(text)


def test_build_sql_includes_sex_when_column_present() -> None:
    cq = parse_cohort_query("mujeres diabéticas en la cohorte")
    spec = cohort_query_to_nl_spec(cq)
    assert spec.sex == "F"
    sql, warns = build_sql_from_cohort(
        cq,
        patient_cols={"id", "birthdate", "deathdate", "gender"},
        condition_cols={"patient", "description"},
        medication_cols=None,
    )
    assert sql
    assert "gender" in sql.lower() or "Gender" in sql
    assert "'F'" in sql or '"F"' in sql or "= 'F'" in sql
    assert not any("ignorada" in w for w in warns)


def test_build_sql_sex_warning_without_gender_column() -> None:
    cq = parse_cohort_query("hombres con diabetes")
    sql, warns = build_sql_from_cohort(
        cq,
        patient_cols={"id", "birthdate", "deathdate"},
        condition_cols={"patient", "description"},
        medication_cols=None,
    )
    assert sql
    assert any("sexo" in w.lower() for w in warns)


def test_dynamic_vocab_matches_semaglutide(tmp_path) -> None:
    db = tmp_path / "term.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE conditions (description TEXT)")
    conn.execute(
        "INSERT INTO conditions VALUES ('Injection semaglutide 1 MG/0.5 ML weekly'),"
        "('Hypertension disorder')"
    )
    conn.execute("CREATE TABLE medications (description TEXT)")
    conn.execute("INSERT INTO medications VALUES ('Semaglutide 1 MG/0.5 ML solution')")
    conn.commit()
    conn.close()

    with sqlite3.connect(str(db)) as c1:
        kc = frozenset(load_known_conditions(c1))
    with sqlite3.connect(str(db)) as c2:
        km = frozenset(load_known_medications(c2))
    clear_terminology_cache()

    cq = parse_cohort_query(
        "cuántos pacientes con semaglutide en tratamiento",
        known_condition_terms=kc,
        known_medication_terms=km,
    )
    assert "semaglutide" in cq.medication_like_tokens or "semaglutide" in cq.condition_like_tokens

    cq2 = parse_cohort_query(
        "pacientes con semaglut en tratamiento",
        known_condition_terms=kc,
        known_medication_terms=km,
    )
    assert "semaglutide" in cq2.medication_like_tokens or "semaglutide" in cq2.condition_like_tokens


def test_parse_diabeticos_gt_65() -> None:
    cq = parse_cohort_query("diabéticos >65")
    assert cq.age_min_years == 65
    assert "diabet" in cq.condition_like_tokens


def test_vocab_match_still_merges_bootstrap_second_condition(tmp_path) -> None:
    """Si el lexicon solo recoge una condición, el bootstrap sigue añadiendo otras del NL (p. ej. hipertensión)."""
    db = tmp_path / "term.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE conditions (description TEXT)")
    conn.execute("INSERT INTO conditions VALUES ('Type 2 diabetes mellitus')")
    conn.commit()
    conn.close()
    with sqlite3.connect(str(db)) as c1:
        kc = frozenset(load_known_conditions(c1))
    clear_terminology_cache()
    cq = parse_cohort_query(
        "pacientes con diabetes e hipertension mayores de 65",
        known_condition_terms=kc,
        known_medication_terms=None,
    )
    assert "diabet" in cq.condition_like_tokens
    assert "hypertens" in cq.condition_like_tokens
    assert cq.age_min_years == 65


def test_like_tokens_for_display_drops_prefix_redundant() -> None:
    from app.capabilities.clinical_sql.cohort_parser import like_tokens_for_display

    assert like_tokens_for_display(["diabet", "diabetes"]) == ["diabetes"]
    assert like_tokens_for_display(["diabetes", "diabet"]) == ["diabetes"]
    assert like_tokens_for_display(["hypertens", "hypertension"]) == ["hypertension"]
    assert like_tokens_for_display(["Diabetes", "diabet"]) == ["Diabetes"]
    assert like_tokens_for_display(["asma", "diabetes"]) == ["asma", "diabetes"]
    assert like_tokens_for_display(["metform", "metformin"]) == ["metformin"]
    assert like_tokens_for_display(["diabet"]) == ["diabet"]
    assert like_tokens_for_display([]) == []


def test_humanize_like_tokens_es_maps_common_tokens() -> None:
    from app.capabilities.clinical_sql.cohort_parser import humanize_like_tokens_es

    assert humanize_like_tokens_es(["diabetes", "hypertens"]) == ["diabetes", "hipertensión"]
    assert humanize_like_tokens_es(["metform", "statin"]) == ["metformina", "estatinas"]
    assert humanize_like_tokens_es(["unknowntokenxyz"]) == ["Unknowntokenxyz"]


def test_sql_synonym_tokens_use_or_inside_single_exists() -> None:
    from app.capabilities.clinical_sql.cohort_nl import CohortNLSpec, build_synthea_cohort_count_sql

    spec = CohortNLSpec(condition_like_tokens=("diabet", "diabetes"))
    sql, _ = build_synthea_cohort_count_sql(
        patient_cols={"id", "birthdate", "deathdate"},
        condition_cols={"patient", "description"},
        medication_cols=None,
        spec=spec,
    )
    assert sql
    assert " OR " in sql
    assert " LIKE " in sql
    assert sql.lower().count("exists (select 1 from conditions") == 1


def test_parse_drops_stroke_when_prevencion_de_ictus_sin_antecedente() -> None:
    cq = parse_cohort_query(
        "Pacientes con diabetes mayores de 75 años, anticoagulación para prevención de ictus"
    )
    assert "stroke" not in cq.condition_like_tokens
    assert "diabet" in cq.condition_like_tokens


def test_parse_keeps_stroke_when_antecedente_de_ictus() -> None:
    cq = parse_cohort_query("Pacientes con diabetes mellitus y antecedente de ictus mayores de 70")
    assert "stroke" in cq.condition_like_tokens


def test_parse_drops_mellitus_when_diabetes_mellitus_vocab() -> None:
    cq = parse_cohort_query(
        "Pacientes con diabetes mellitus tipo 2 e hipertensión",
        known_condition_terms=frozenset({"mellitus", "diabetes mellitus", "hypertension"}),
    )
    assert "mellitus" not in cq.condition_like_tokens
    assert "diabet" in cq.condition_like_tokens or "diabetes" in cq.condition_like_tokens


def test_parse_drops_renal_when_solo_funcion_renal() -> None:
    cq = parse_cohort_query(
        "Pacientes con diabetes, hipertensión y función renal a valorar antes de anticoagular",
        known_condition_terms=frozenset({"renal", "diabetes", "hypertension"}),
    )
    assert "renal" not in cq.condition_like_tokens
