"""Recuperación PubMed condicionada por desenlaces CV + priors de diseño."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_intent import extract_clinical_intent, intent_asks_cv_outcomes
from app.capabilities.evidence_rag.evidence_rerank import (
    clinical_weak_evidence_share,
    infer_study_type_from_title,
    study_design_weight,
)
from app.capabilities.evidence_rag.heuristic_evidence_query import (
    build_evidence_retrieval_stages,
    build_evidence_search_queries,
)
from app.schemas.copilot_state import ClinicalContext

_HERO = (
    "En pacientes con diabetes e hipertensión ≥65 años, "
    "¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 "
    "frente a solo metformina para reducir eventos cardiovasculares?"
)


def test_hero_broad_primary_stages_with_elderly() -> None:
    ctx = ClinicalContext(
        population_conditions=["diabet", "hipertens"],
        population_age_min=65,
    )
    stages = build_evidence_retrieval_stages(_HERO, ctx)
    ids = [s.stage_id for s in stages]
    assert ids[0] == "broad_primary"
    assert "cvot_landmark" in ids
    q = stages[0].query.lower()
    assert "aged" in q or "elderly" in q or "older adult" in q
    assert q.count(" or ") <= 20


def test_safety_intent_adds_adverse_events_block() -> None:
    q = "Is semaglutide safe in type 2 diabetes? adverse events and hypoglycemia"
    ctx = ClinicalContext(population_conditions=["diabet"])
    stages = build_evidence_retrieval_stages(q, ctx)
    assert stages and stages[0].stage_id.startswith("safety")
    assert "adverse events" in stages[0].query.lower()


def test_hero_cv_intent_broad_query_has_core_axes() -> None:
    ctx = ClinicalContext(
        population_conditions=["diabet", "hipertens"],
        population_age_min=65,
    )
    intent = extract_clinical_intent(_HERO, ctx)
    assert intent_asks_cv_outcomes(intent)
    queries = build_evidence_search_queries(_HERO, ctx)
    assert len(queries) >= 1
    q0 = queries[0].lower()
    assert "sglt2" in q0 or "gliflozin" in q0
    assert "cardiovascular" in q0 or "mace" in q0
    assert "diabet" in q0 or "type 2 diabetes" in q0
    assert q0.count(" or ") <= 20


def test_cvot_trial_title_inference() -> None:
    assert infer_study_type_from_title("EMPA-REG OUTCOME: cardiovascular outcomes in T2DM") == (
        "cvot-outcomes-trial"
    )


def test_study_design_weight_cv_intent_boosts_cvot() -> None:
    intent = extract_clinical_intent(_HERO)
    assert study_design_weight("cvot-outcomes-trial", clinical_intent=intent) == 1.0
    assert study_design_weight("narrative-review", clinical_intent=intent) == 0.45


def test_clinical_weak_evidence_share_detects_depression_noise() -> None:
    intent = extract_clinical_intent(_HERO)
    share = clinical_weak_evidence_share(
        [
            (
                "GLP-1 and depression in type 2 diabetes: a review",
                "depression outcomes without cardiovascular events",
            ),
            (
                "DECLARE-TIMI 58: cardiovascular outcomes with dapagliflozin",
                "MACE and cardiovascular death in type 2 diabetes",
            ),
        ],
        clinical_intent=intent,
    )
    assert share >= 0.5
