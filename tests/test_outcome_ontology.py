"""Ontología de desenlaces y memoria clínica landmark."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_intent import extract_clinical_intent
from app.capabilities.evidence_rag.clinical_knowledge import (
    LANDMARK_CVOTS,
    landmark_pubmed_acronyms_clause,
    landmark_rerank_boost,
    match_landmark_trial,
)
from app.capabilities.evidence_rag.heuristic_evidence_query import build_evidence_retrieval_stages
from app.capabilities.evidence_rag.noise_suppression import (
    detect_negative_topics,
    detect_negative_topics_for_intent,
)
from app.capabilities.evidence_rag.outcome_ontology import (
    graded_cv_outcome_score,
    pubmed_clause_cv_strict,
)
from app.schemas.copilot_state import ClinicalContext

_HERO = (
    "En pacientes con diabetes e hipertensión ≥65 años, "
    "¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 "
    "frente a solo metformina para reducir eventos cardiovasculares?"
)


def test_mace_ontology_expands_components_in_pubmed_clause() -> None:
    q = pubmed_clause_cv_strict().lower()
    assert "nonfatal myocardial infarction" in q
    assert "cardiovascular death" in q
    assert "nonfatal stroke" in q
    assert "mace" in q


def test_graded_cv_outcome_score_mace_components() -> None:
    blob = "cardiovascular death and nonfatal myocardial infarction in type 2 diabetes"
    assert graded_cv_outcome_score(blob) >= 0.4


def test_landmark_match_empa_reg() -> None:
    trial = match_landmark_trial(
        "EMPA-REG OUTCOME: cardiovascular outcomes with empagliflozin",
        "cardiovascular death in patients with type 2 diabetes",
    )
    assert trial is not None
    assert "empagliflozin" in trial.drugs


def test_landmark_rerank_boost_positive() -> None:
    intent = extract_clinical_intent(_HERO)
    boost = landmark_rerank_boost(
        "LEADER trial liraglutide cardiovascular outcomes",
        "MACE reduction",
        clinical_intent=intent,
    )
    assert boost >= 0.15


def test_noise_suppression_pregnancy_penalized_unless_intent() -> None:
    blob = "pregnancy outcomes with metformin safety"
    assert "pregnancy" in detect_negative_topics(blob)
    from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent

    intent = ClinicalIntent(population=["pregnancy"])
    filtered = detect_negative_topics_for_intent(blob, clinical_intent=intent)
    assert "pregnancy" not in filtered


def test_cv_stages_include_broad_primary_and_landmark() -> None:
    ctx = ClinicalContext(
        population_conditions=["diabet", "hipertens"],
        population_age_min=65,
    )
    stages = build_evidence_retrieval_stages(_HERO, ctx)
    ids = [s.stage_id for s in stages]
    assert ids[0] == "broad_primary"
    assert "cvot_landmark" in ids


def test_landmark_knowledge_base_not_empty() -> None:
    assert len(LANDMARK_CVOTS) >= 8
    assert landmark_pubmed_acronyms_clause()
