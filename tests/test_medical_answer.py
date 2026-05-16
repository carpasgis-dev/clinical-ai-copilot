"""Fase 3.4 — respuesta médica estructurada (stub) y render legado."""
from __future__ import annotations

from app.api.schemas import MedicalAnswer
from app.orchestration.medical_answer_builder import (
    build_stub_medical_answer,
    render_medical_answer_to_text,
)
from app.schemas.copilot_state import ClinicalContext, Route


def test_build_stub_medical_answer_hybrid_shape() -> None:
    state = {
        "route": Route.HYBRID,
        "user_query": "demo",
        "sql_result": {"rows": [{"cohort_size": 2}], "row_count": 2},
        "clinical_context": ClinicalContext(
            population_conditions=["diabet"],
            population_medications=["metform"],
            population_age_min=65,
        ).model_dump(mode="json"),
        "evidence_bundle": {
            "articles": [
                {
                    "pmid": "123",
                    "title": "Example trial",
                    "year": 2020,
                    "abstract_snippet": "Pacientes con diabetes tipo 2 mostraron reducción de HbA1c.",
                },
            ],
            "pmids": ["123"],
        },
    }
    ma = build_stub_medical_answer(state)
    v = MedicalAnswer.model_validate(ma)
    assert v.cohort_size == 2
    assert v.cohort_summary and "cohorte local" in v.cohort_summary.lower()
    assert v.evidence_summary and "referencias" in v.evidence_summary.lower()
    assert "demo" in (v.evidence_summary or "")
    assert len(v.citations) >= 1
    assert v.citations[0].pmid == "123"
    assert v.evidence_statements
    assert v.evidence_statements[0].citation_pmids == ["123"]
    assert "123" in v.evidence_statements[0].statement
    assert "HbA1c" in v.evidence_statements[0].statement
    assert v.key_findings
    assert v.recommendations
    assert len(v.limitations) >= 1


def test_build_stub_medical_answer_sql_only() -> None:
    state = {
        "route": Route.SQL,
        "user_query": "q",
        "sql_result": {"rows": [{"cohort_size": 5}], "row_count": 5},
    }
    ma = build_stub_medical_answer(state)
    v = MedicalAnswer.model_validate(ma)
    assert v.cohort_size == 5
    assert v.cohort_summary and "5" in v.cohort_summary
    assert v.evidence_summary is None
    assert v.citations == []
    assert v.evidence_statements == []


def test_build_stub_medical_answer_sql_only() -> None:
    ma = build_stub_medical_answer(
        {
            "route": Route.SQL,
            "user_query": "q",
            "sql_result": {"rows": [{"cohort_size": 1}], "row_count": 1},
        }
    )
    text = render_medical_answer_to_text(ma)
    assert "[synthesis stub]" not in text.lower()
    assert "cohorte local" in text.lower()
