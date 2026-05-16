"""Stage 1: Clinical Intent Graph y políticas downstream."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_intent_graph import (
    build_clinical_intent_graph,
    infer_expected_landmark_trials,
    landmarks_found_in_articles,
)
from app.capabilities.evidence_rag.evidence_policy import (
    article_matches_suppress_policy,
    pubmed_noise_exclusion_clause,
)
from app.capabilities.evidence_rag.heuristic_evidence_query import build_evidence_retrieval_stages
from app.schemas.copilot_state import ClinicalContext

_DOAC_QUERY = (
    "En pacientes con fibrilación auricular y diabetes, "
    "¿DOAC o warfarina para prevención de ictus y riesgo de hemorragia?"
)

_HERO = (
    "En pacientes con diabetes e hipertensión ≥65 años, "
    "¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 "
    "frente a solo metformina para reducir eventos cardiovasculares?"
)


def test_doac_comparative_graph_structure() -> None:
    ctx = ClinicalContext(
        population_conditions=["diabet", "fibrilacion"],
        population_medications=["warfarin"],
    )
    graph = build_clinical_intent_graph(_DOAC_QUERY, ctx)
    assert graph.question_type == "comparative_effectiveness"
    assert "atrial fibrillation" in graph.population
    assert any("warfarin" in c.lower() for c in graph.comparator)
    assert any(
        x.lower() in ("doac", "apixaban", "rivaroxaban", "anticoagulation")
        for x in graph.intervention
    )
    assert "stroke" in graph.outcomes or "systemic embolism" in graph.outcomes
    assert "prediction_model" in graph.suppress_evidence
    assert "ARISTOTLE" in graph.expected_landmark_trials


def test_anticoag_retrieval_stages_not_generic_af_noise() -> None:
    graph = build_clinical_intent_graph(_DOAC_QUERY)
    stages = build_evidence_retrieval_stages(_DOAC_QUERY)
    assert stages
    assert stages[0].stage_id == "anticoag_comparative_primary"
    q = stages[0].query.lower()
    assert "warfarin" in q or "vitamin k" in q
    assert "not (" in q
    assert any(s.stage_id == "anticoag_landmark" for s in stages)


def test_pubmed_noise_exclusion_for_comparative() -> None:
    graph = build_clinical_intent_graph(_DOAC_QUERY)
    clause = pubmed_noise_exclusion_clause(graph)
    assert "NOT (" in clause
    assert "telemedicine" in clause.lower() or "machine learning" in clause.lower()


def test_telemedicine_suppressed_by_policy() -> None:
    graph = build_clinical_intent_graph(_DOAC_QUERY)
    tit = "E-health integration in cardiology: ANMCO position paper"
    snip = "Telemedicine and digital health for atrial fibrillation monitoring."
    assert article_matches_suppress_policy(tit, snip, graph)


def test_landmark_detection_in_pool() -> None:
    expected = infer_expected_landmark_trials(build_clinical_intent_graph(_DOAC_QUERY))
    arts = [
        {
            "title": "Apixaban versus warfarin in patients with atrial fibrillation (ARISTOTLE)",
            "abstract_snippet": "Stroke and major bleeding.",
        }
    ]
    found = landmarks_found_in_articles(arts, expected)
    assert any("ARISTOTLE" in f for f in found)


def test_hero_still_uses_cv_broad_ladder() -> None:
    ctx = ClinicalContext(population_conditions=["diabet", "hipertens"], population_age_min=65)
    stages = build_evidence_retrieval_stages(_HERO, ctx)
    ids = [s.stage_id for s in stages]
    assert "broad_primary" in ids
    assert "anticoag_comparative_primary" not in ids


def test_graph_roundtrip_dict() -> None:
    g = build_clinical_intent_graph(_DOAC_QUERY)
    d = g.to_dict()
    from app.capabilities.evidence_rag.clinical_intent_graph import ClinicalIntentGraph

    g2 = ClinicalIntentGraph.from_dict(d)
    assert g2.question_type == g.question_type
    assert g2.expected_landmark_trials == g.expected_landmark_trials
