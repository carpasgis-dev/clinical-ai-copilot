"""Tests del módulo de razonamiento (fase 3.7)."""
from __future__ import annotations

from app.orchestration.reasoning import build_reasoning_state
from app.schemas.copilot_state import ArticleSummary, EvidenceBundle, Route


def test_reasoning_reads_evidence_bundle_as_pydantic_model() -> None:
    """El estado LangGraph puede tener ``EvidenceBundle`` como modelo, no solo dict."""
    bundle = EvidenceBundle(
        search_term="test",
        pmids=["123"],
        articles=[
            ArticleSummary(pmid="123", title="RCT of diabetes care", year=2023),
        ],
    )
    state = {
        "route": Route.HYBRID,
        "sql_result": None,
        "clinical_context": None,
        "evidence_bundle": bundle,
    }
    rs = build_reasoning_state(state)
    assert len(rs.evidence_assessments) == 1
    assert rs.evidence_assessments[0].pmid == "123"
    assert not rs.conflicts


def test_hero_query_router_reason_includes_sql_signal() -> None:
    from app.orchestration.router import classify_route

    hero = (
        "Paciente con diabetes e hipertensión mayor de 65 años. "
        "¿Qué evidencia reciente existe sobre tratamientos que "
        "reduzcan riesgo cardiovascular?"
    )
    route, reason = classify_route(hero)
    assert route == Route.HYBRID
    assert "sql=" in reason and "evidence=" in reason
    # Al menos una señal SQL (paciente con / mayor de / …), no solo evidence=1.
    import re

    m = re.search(r"sql=(\d+)", reason)
    assert m is not None and int(m.group(1)) >= 1


def test_reasoning_empty_hits_with_pubmed_term_is_uncertainty_not_conflict() -> None:
    """0 artículos tras construir query: incertidumbre operativa, no 'conflicto' clínico."""
    state = {
        "route": Route.HYBRID,
        "pubmed_query": '("diabetes"[tiab]) AND therapy',
        "sql_result": None,
        "clinical_context": None,
        "evidence_bundle": EvidenceBundle(search_term="x", pmids=[], articles=[]).model_dump(),
    }
    rs = build_reasoning_state(state)
    assert not rs.conflicts
    assert rs.uncertainty_notes
    assert "PubMed se consultó" in rs.uncertainty_notes[0]
    assert rs.evidence_quality == "sin_resultados_pubmed"


def test_reasoning_cohort_summary_lists_clinical_filters() -> None:
    state = {
        "route": Route.HYBRID,
        "sql_result": {"rows": [{"cohort_size": 10}]},
        "clinical_context": {
            "population_conditions": ["diabetes", "hipertensión"],
            "population_medications": [],
            "population_age_min": 65,
            "population_sex": None,
        },
        "evidence_bundle": EvidenceBundle(search_term="x", pmids=[], articles=[]).model_dump(),
    }
    rs = build_reasoning_state(state)
    assert rs.cohort_summary
    assert "n≈10" in rs.cohort_summary
    assert "diabetes" in rs.cohort_summary
    assert "hipertensión" in rs.cohort_summary
    assert "edad ≥ 65" in rs.cohort_summary


def test_reasoning_evidence_assessment_applicability_with_elderly_cohort() -> None:
    """Con cohorte ≥65 y título en adolescentes, ``applicability`` no debe quedar vacío."""
    state = {
        "route": Route.HYBRID,
        "sql_result": {"rows": [{"cohort_size": 10}]},
        "clinical_context": {
            "population_conditions": ["diabetes", "hipertensión"],
            "population_age_min": 65,
        },
        "evidence_bundle": EvidenceBundle(
            search_term="x",
            pmids=["999"],
            articles=[
                ArticleSummary(
                    pmid="999",
                    title="Metabolic outcomes in adolescents with PCOS: a meta-analysis",
                    year=2024,
                ),
            ],
        ).model_dump(),
    }
    rs = build_reasoning_state(state)
    assert len(rs.evidence_assessments) == 1
    assert rs.evidence_assessments[0].applicability
    assert rs.evidence_assessments[0].applicability.startswith("Limitada")
    assert rs.applicability_notes


def test_reasoning_includes_synthesis_calibration_from_state() -> None:
    """``build_reasoning_state`` propaga calibración calculada en el nodo previo."""
    state = {
        "route": Route.HYBRID,
        "sql_result": {"rows": [{"cohort_size": 2}]},
        "clinical_context": {"population_conditions": ["diabetes"]},
        "synthesis_calibration": {
            "retrieval_outcome": "partial_primary_miss",
            "dominant_retrieval_tier": 4,
            "retrieval_confidence": 0.35,
            "evidence_specificity": 0.2,
            "applicability_confidence": 0.5,
            "primary_stage_hit": False,
            "landmark_present": False,
        },
        "evidence_bundle": EvidenceBundle(
            search_term="x",
            pmids=["111"],
            articles=[
                {
                    "pmid": "111",
                    "title": "SGLT2 inhibitors cardiovascular outcomes review",
                    "abstract_snippet": "MACE cardiovascular mortality",
                    "retrieval_tier": 4,
                    "retrieval_stage": "broad_relaxed",
                },
            ],
        ).model_dump(),
    }
    rs = build_reasoning_state(state)
    assert rs.synthesis_calibration is not None
    assert rs.synthesis_calibration["retrieval_outcome"] == "partial_primary_miss"
    assert rs.synthesis_calibration["dominant_retrieval_tier"] == 4
    assert any("Calibración síntesis" in n for n in rs.uncertainty_notes)
    assert any("baja confianza" in n.lower() for n in rs.uncertainty_notes)


def test_query_response_reasoning_exposes_synthesis_calibration() -> None:
    from app.api.response_format import build_query_response

    state = {
        "route": "hybrid",
        "route_reason": "test",
        "final_answer": "ok",
        "reasoning_state": {
            "cohort_summary": "n≈2",
            "evidence_assessments": [],
            "uncertainty_notes": [],
            "conflicts": [],
            "applicability_notes": [],
            "evidence_quality": "mixta",
            "synthesis_calibration": {
                "retrieval_outcome": "success",
                "dominant_retrieval_tier": 2,
                "retrieval_confidence": 1.0,
                "evidence_specificity": 0.8,
                "applicability_confidence": 0.7,
                "primary_stage_hit": True,
                "landmark_present": True,
            },
        },
    }
    resp = build_query_response(state, session_id="test", latency_ms=1.0)
    assert resp.reasoning_state is not None
    cal = resp.reasoning_state.synthesis_calibration
    assert cal is not None
    assert cal.retrieval_outcome == "success"
    assert cal.dominant_retrieval_tier == 2
    assert cal.retrieval_confidence == 1.0
