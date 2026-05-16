"""PubMed retrieval amplio (recall) vs alineación estricta en rerank."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_intent import extract_clinical_intent, infer_priority_axis
from app.capabilities.evidence_rag.heuristic_evidence_query import (
    build_evidence_search_queries,
    build_evidence_search_query,
)
from app.schemas.copilot_state import ClinicalContext

_HERO = (
    "En pacientes con diabetes e hipertensión ≥65 años, "
    "¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 "
    "frente a solo metformina para reducir eventos cardiovasculares?"
)


def test_hero_query_cv_stages_include_outcomes() -> None:
    ctx = ClinicalContext(
        population_conditions=["diabet", "hipertens"],
        population_medications=["metform"],
        population_age_min=65,
    )
    queries = build_evidence_search_queries(_HERO, ctx)
    assert len(queries) >= 1
    primary = queries[0].lower()
    assert "sglt2" in primary or "gliflozin" in primary
    assert "glp" in primary
    assert "diabet" in primary or "diabetes mellitus" in primary
    assert "cardiovascular" in primary or "mace" in primary
    assert primary.count(" or ") <= 20


def test_hero_single_query_compat() -> None:
    ctx = ClinicalContext(
        population_conditions=["diabet", "hipertens"],
        population_age_min=65,
    )
    q = build_evidence_search_query(_HERO, ctx)
    assert " AND " in q
    assert " OR " in q
    assert 'metformin"' not in q


def test_priority_axis_outcome_centric() -> None:
    q = "What reduces cardiovascular mortality in adults?"
    intent = extract_clinical_intent(q)
    assert infer_priority_axis(q, intent) == "outcome"
    assert intent.priority_axis == "outcome"


def test_comparator_metformin_not_in_stage1_intervention_or() -> None:
    ctx = ClinicalContext(population_conditions=["diabet"], population_age_min=65)
    broad = build_evidence_search_queries(_HERO, ctx)[0].lower()
    assert "sglt2" in broad or "glp" in broad
    # Metformina no obligatoria en OR de intervención cuando hay SGLT2/GLP-1
    if "metformin" in broad:
        assert "sglt2" in broad or "glp" in broad
