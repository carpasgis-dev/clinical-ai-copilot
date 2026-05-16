"""Noise suppression, intent semantic query y modo semántico off (sin torch)."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_intent import extract_clinical_intent
from app.capabilities.evidence_rag.evidence_rerank import rerank_article_dicts
from app.capabilities.evidence_rag.intent_semantic_query import (
    build_intent_semantic_query,
    preferred_study_types_for_intent,
)
from app.capabilities.evidence_rag.noise_suppression import (
    apply_noise_suppression,
    topic_drift_multiplier,
)
from app.capabilities.evidence_rag.semantic_config import (
    cross_encoder_model_name,
    embedding_model_name,
    semantic_rerank_mode,
    semantic_score_weights,
)
from app.schemas.copilot_state import ClinicalContext

_HERO = (
    "En pacientes con diabetes e hipertensión ≥65 años, "
    "¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 "
    "frente a solo metformina para reducir eventos cardiovasculares?"
)


def test_semantic_mode_default_off() -> None:
    import os

    # En test, si COPILOT_SEMANTIC_RERANK no está en env, debe ser off.
    original = os.environ.pop("COPILOT_SEMANTIC_RERANK", None)
    try:
        assert semantic_rerank_mode() == "off"
    finally:
        if original is not None:
            os.environ["COPILOT_SEMANTIC_RERANK"] = original


def test_semantic_config_biomedical_defaults() -> None:
    import os
    # Sin overrides de env, los defaults son modelos biomédicos
    for var in ("COPILOT_EMBEDDING_MODEL", "COPILOT_RERANKER_MODEL"):
        os.environ.pop(var, None)
    assert "bge-large" in embedding_model_name() or "MedCPT" in embedding_model_name()
    assert "MedCPT" in cross_encoder_model_name() or "reranker" in cross_encoder_model_name()
    w_h, w_e, w_c = semantic_score_weights()
    # cross-encoder debe dominar (>50% del peso total)
    assert w_c > w_h + w_e


def test_intent_semantic_query_hero_english() -> None:
    ctx = ClinicalContext(population_conditions=["diabet", "hipertens"], population_age_min=65)
    intent = extract_clinical_intent(_HERO, ctx)
    q = build_intent_semantic_query(intent, _HERO).lower()
    assert "type 2 diabetes" in q or "diabetes" in q
    assert "sglt2" in q or "glp-1" in q
    assert "metformin" in q
    assert "mace" in q or "cardiovascular" in q
    prefs = preferred_study_types_for_intent(intent)
    assert "cvot" in prefs


def test_noise_suppression_structural_penalty() -> None:
    intent = extract_clinical_intent(_HERO)
    mult = topic_drift_multiplier(
        "SGLT2 in murine model",
        "animal model without human outcomes",
        clinical_intent=intent,
    )
    assert mult == 0.25
    score, ns = apply_noise_suppression(1.0, "In vitro study of SGLT2", "zebrafish cell line", clinical_intent=intent)
    assert score == 0.25
    assert ns.translational_penalty == 0.25
    assert ns.evidence_penalty == 1.0
    assert ns.matched_topics


def test_noise_suppression_differentiated_evidence_weak() -> None:
    from app.capabilities.evidence_rag.noise_suppression import compute_structural_noise

    assert compute_structural_noise(["commentary"]).multiplier == 0.82
    assert compute_structural_noise(["editorial"]).multiplier == 0.75
    assert compute_structural_noise(["case report"]).multiplier == 0.45


def test_noise_suppression_double_translational_not_extra_harsh() -> None:
    from app.capabilities.evidence_rag.noise_suppression import compute_structural_noise

    single = compute_structural_noise(["preclinical"]).multiplier
    double = compute_structural_noise(["preclinical", "rodent model"]).multiplier
    assert single == 0.25
    assert double == 0.25


def test_rerank_off_uses_noise_on_oncology_drift() -> None:
    intent = extract_clinical_intent(_HERO)
    arts = [
        {
            "pmid": "1",
            "title": "DECLARE-TIMI 58 cardiovascular outcomes dapagliflozin type 2 diabetes",
            "abstract_snippet": "MACE cardiovascular death heart failure",
        },
        {
            "pmid": "2",
            "title": "Breast cancer and implant failure perioperative fracture ORIF",
            "abstract_snippet": "oncology orthopedic murine model",
        },
    ]
    ranked = rerank_article_dicts(arts, _HERO, cap=2, clinical_intent=intent)
    assert ranked[0]["pmid"] == "1"
