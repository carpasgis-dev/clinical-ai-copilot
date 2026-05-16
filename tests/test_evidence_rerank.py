"""Re-ranking post-retrieval: relevancia + jerarquía por diseño inferido del título."""
from __future__ import annotations

import pytest

from app.capabilities.evidence_rag.evidence_rerank import (
    cohort_lexical_adjustment,
    infer_applicability_line,
    infer_study_type_from_title,
    rerank_article_dicts,
    weak_design_share_from_titles,
)


def test_weak_design_share_from_titles() -> None:
    titles = [
        "Acute event: a case report",
        "Another patient: a case report",
        "Network meta-analysis of drug X",
    ]
    assert weak_design_share_from_titles(titles) == pytest.approx(2 / 3)
    assert weak_design_share_from_titles([]) == 0.0


def test_infer_applicability_adolescent_vs_cohort_65() -> None:
    t = "PCOS and metabolic syndrome in adolescents: a systematic review and meta-analysis."
    line = infer_applicability_line(
        t,
        population_age_min=65,
        population_conditions=["diabetes"],
    )
    assert line is not None
    assert line.startswith("Limitada")


def test_cohort_lexical_penalizes_sarcoidosis_without_diabetes_context() -> None:
    adj = cohort_lexical_adjustment(
        "Cardiac sarcoidosis trial design",
        "interleukin granuloma arrhythmia",
        population_conditions=["diabetes", "hipertensión"],
        population_medications=None,
    )
    assert adj < -0.05


def test_rerank_demotes_adolescent_meta_when_cohort_elderly() -> None:
    uq = "diabetes hipertensión mayores 65 años tratamiento riesgo cardiovascular"
    arts = [
        {
            "pmid": "1",
            "title": "PCOS and metabolic syndrome in adolescents: meta-analysis",
            "abstract_snippet": "adolescents with PCOS",
        },
        {
            "pmid": "2",
            "title": "Resistance training in older adults with type 2 diabetes: systematic review",
            "abstract_snippet": "older adults type 2 diabetes",
        },
    ]
    out = rerank_article_dicts(
        arts,
        uq,
        cap=2,
        population_age_min=65,
        population_conditions=["diabetes", "hipertensión"],
    )
    assert out[0]["pmid"] == "2"


def test_infer_study_type_case_report() -> None:
    t = "Acute kidney injury in an elderly patient: A case report."
    assert infer_study_type_from_title(t) == "case-report"


def test_infer_study_type_meta_analysis() -> None:
    t = "Network meta-analysis of antihypertensive agents in diabetes."
    assert infer_study_type_from_title(t) in ("network-meta-analysis", "meta-analysis")


def test_rerank_prefers_meta_over_case_report_for_treatment_query() -> None:
    uq = "¿Qué tratamientos reducen riesgo cardiovascular en diabetes tipo 2?"
    arts = [
        {
            "pmid": "1",
            "title": "Single patient hypotension: a case report",
            "abstract_snippet": "We describe one patient with diabetes.",
        },
        {
            "pmid": "2",
            "title": "Network meta-analysis of GLP-1 agonists and cardiovascular outcomes in type 2 diabetes",
            "abstract_snippet": "Randomized trials were pooled comparing GLP-1 receptor agonists.",
        },
    ]
    out = rerank_article_dicts(arts, uq, cap=2)
    assert out[0]["pmid"] == "2"
    assert out[1]["pmid"] == "1"


def test_rerank_demotes_hiv_art_protocol_when_query_is_dm_htn_elderly() -> None:
    """RCT con VIH/ART no debe encabezar si la petición no invoca VIH (ruta híbrida metabólica)."""
    uq = (
        "Paciente con diabetes e hipertensión mayor de 65 años. "
        "¿Qué evidencia reciente existe sobre tratamientos que reduzcan riesgo cardiovascular?"
    )
    hiv_rct = {
        "pmid": "42050587",
        "title": (
            "Optimizing metabolic management on integrase-based ART (OPTIMAR): study protocol "
            "for people with HIV on integrase strand transfer inhibitor-based antiretroviral therapy."
        ),
        "abstract_snippet": "People with HIV have an increased cardiovascular disease risk.",
    }
    cohort_young = {
        "pmid": "42044147",
        "title": (
            "Combined Pre-Hypertension, Pre-Diabetes, and Pre-Hyperlipidemia and Long-Term "
            "Cardiovascular Risk in Young Adults: A Nationwide Cohort Study."
        ),
        "abstract_snippet": "Young adults with pre-hypertension, pre-diabetes, and pre-hyperlipidemia.",
    }
    arts = [hiv_rct, cohort_young]
    out = rerank_article_dicts(
        arts,
        uq,
        cap=2,
        population_age_min=65,
        population_conditions=["diabetes", "hipertensión"],
    )
    assert out[0]["pmid"] == "42044147"
    assert out[1]["pmid"] == "42050587"


def test_infer_applicability_limitada_vih_sin_mencion_en_pregunta() -> None:
    line = infer_applicability_line(
        "Cardiometabolic outcomes in people with HIV on integrase-based ART.",
        population_age_min=65,
        population_conditions=["diabetes", "hipertensión"],
        user_query="diabetes hipertensión adultos mayores riesgo cardiovascular",
        abstract_snippet="",
    )
    assert line is not None
    assert line.startswith("Limitada")
    assert "VIH" in line or "vih" in line.lower()
