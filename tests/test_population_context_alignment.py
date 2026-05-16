"""Alineación de nichos clínicos (pregunta/cohorte ↔ artículo)."""
from __future__ import annotations

from app.capabilities.evidence_rag.population_context_alignment import (
    niche_applicability_limitada_line,
    niche_mismatch_penalty,
)


def test_niche_applicability_limitada_para_vih_sin_invitacion_en_contexto() -> None:
    tit = (
        "Optimizing metabolic management on integrase-based ART (OPTIMAR): study protocol "
        "for people with HIV on integrase strand transfer inhibitor-based antiretroviral therapy."
    )
    line = niche_applicability_limitada_line(
        tit,
        "",
        user_query="diabetes hipertensión mayores 65 años riesgo cardiovascular",
        population_conditions=["diabetes", "hipertensión"],
        population_medications=None,
    )
    assert line is not None
    assert line.startswith("Limitada")
    assert "VIH" in line or "vih" in line.lower()

    no_line = niche_applicability_limitada_line(
        tit,
        "",
        user_query="VIH diabetes riesgo cardiovascular",
        population_conditions=["diabetes"],
        population_medications=None,
    )
    assert no_line is None


def test_penaliza_vih_si_contexto_no_menciona_vih() -> None:
    p = niche_mismatch_penalty(
        "Cardiometabolic risk in people with HIV",
        "antiretroviral therapy naive cohort",
        user_query="diabetes hipertensión riesgo cardiovascular",
        population_conditions=["diabetes", "hipertensión"],
        population_medications=None,
    )
    assert p <= -0.1


def test_no_penaliza_vih_si_usuario_lo_pide() -> None:
    p = niche_mismatch_penalty(
        "Cardiometabolic risk in people with HIV",
        "antiretroviral therapy",
        user_query="pacientes con VIH y diabetes mellitus",
        population_conditions=["diabetes"],
        population_medications=None,
    )
    assert p == 0.0


def test_penaliza_embarazo_si_no_aparece_en_contexto() -> None:
    p = niche_mismatch_penalty(
        "Gestational diabetes management",
        "pregnant women randomized trial",
        user_query="diabetes tipo 2 adultos tratamiento",
        population_conditions=["diabetes"],
        population_medications=None,
    )
    assert p <= -0.1


def test_no_penaliza_embarazo_si_contexto_lo_menciona() -> None:
    p = niche_mismatch_penalty(
        "Gestational diabetes",
        "pregnancy outcomes",
        user_query="riesgo en embarazo con diabetes",
        population_conditions=["diabetes gestacional"],
        population_medications=None,
    )
    assert p == 0.0


def test_penaliza_oncologia_si_contexto_no_es_oncologia() -> None:
    p = niche_mismatch_penalty(
        "Solid tumor phase I trial of chemotherapy",
        "metastatic carcinoma patients received radiotherapy",
        user_query="diabetes e hipertension prevencion cardiovascular",
        population_conditions=["diabetes", "hipertensión"],
        population_medications=None,
    )
    assert p <= -0.1
