"""Fusión final de rerank y depriorización de diseños débiles en top-k."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_intent import extract_clinical_intent
from app.capabilities.evidence_rag.evidence_rerank import (
    _demote_weak_evidence_in_topk,
    _final_fusion_score,
    infer_study_type_from_title,
    rerank_article_dicts,
    study_design_weight,
)
from app.capabilities.evidence_rag.heuristic_evidence_query import build_evidence_retrieval_stages
from app.capabilities.evidence_rag.outcome_ontology import pubmed_clause_cv_primary
from app.schemas.copilot_state import ClinicalContext

_HERO = (
    "En pacientes con diabetes e hipertensión ≥65 años, "
    "¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 "
    "frente a solo metformina para reducir eventos cardiovasculares?"
)


def test_primary_cv_clause_excludes_generic_heart_disease() -> None:
    q = pubmed_clause_cv_primary().lower()
    assert "mace" in q or "major adverse cardiovascular" in q
    assert "cvot" in q or "cardiovascular outcome trial" in q
    assert 'heart disease"[tiab]' not in q


def test_hero_broad_primary_uses_tighter_cv_clause() -> None:
    ctx = ClinicalContext(population_conditions=["diabet", "hipertens"], population_age_min=65)
    stages = build_evidence_retrieval_stages(_HERO, ctx)
    primary = stages[0].query.lower()
    assert 'heart disease"[tiab]' not in primary
    assert "mace" in primary or "major adverse cardiovascular" in primary


def test_narrative_review_penalized_vs_cvot_for_cv_intent() -> None:
    intent = extract_clinical_intent(_HERO)
    assert study_design_weight("cvot-outcomes-trial", clinical_intent=intent) > study_design_weight(
        "narrative-review", clinical_intent=intent
    )


def test_final_fusion_prefers_cvot_over_narrative_headline() -> None:
    intent = extract_clinical_intent(_HERO)
    narrative = {
        "pmid": "1",
        "title": "Tirzepatide vs semaglutide: a narrative review",
        "abstract_snippet": "We discuss mechanisms and dosing.",
        "retrieval_tier": 1,
        "rank_score_debug": {"fused": 0.85},
    }
    cvot = {
        "pmid": "2",
        "title": "EMPA-REG OUTCOME: cardiovascular outcomes in type 2 diabetes",
        "abstract_snippet": "Major adverse cardiovascular events and cardiovascular death.",
        "retrieval_tier": 2,
        "rank_score_debug": {"fused": 0.72},
    }
    assert _final_fusion_score(cvot, clinical_intent=intent, clinical_context=None) > _final_fusion_score(
        narrative, clinical_intent=intent, clinical_context=None
    )


def test_demote_narrative_review_from_top3_when_strong_in_pool() -> None:
    intent = extract_clinical_intent(_HERO)
    ranked = [
        {
            "pmid": "42100257",
            "title": "Tirzepatide vs semaglutide for obesity: narrative review",
            "abstract_snippet": "Overview of incretin therapies.",
            "retrieval_tier": 1,
        },
        {
            "pmid": "2",
            "title": "DECLARE-TIMI 58 cardiovascular outcomes",
            "abstract_snippet": "MACE and cardiovascular death in type 2 diabetes.",
            "retrieval_tier": 2,
        },
        {
            "pmid": "3",
            "title": "SGLT2 inhibitors in diabetes",
            "abstract_snippet": "Observational cohort.",
            "retrieval_tier": 1,
        },
    ]
    pool = ranked + [
        {
            "pmid": "4",
            "title": "EMPA-REG OUTCOME trial results",
            "abstract_snippet": "Cardiovascular death and MACE reduction.",
            "retrieval_tier": 2,
        },
    ]
    out = _demote_weak_evidence_in_topk(ranked, pool, cap=3, clinical_intent=intent)
    top_st = infer_study_type_from_title(str(out[0].get("title") or ""))
    assert top_st in ("cvot-outcomes-trial", "meta-analysis", "rct", "systematic-review")
    assert infer_study_type_from_title(str(out[0].get("title") or "")) != "narrative-review"


def test_rerank_article_dicts_applies_final_fusion(monkeypatch) -> None:
    monkeypatch.setenv("COPILOT_SEMANTIC_RERANK", "off")
    intent = extract_clinical_intent(_HERO)
    arts = [
        {
            "pmid": "9",
            "title": "Narrative review of GLP-1 agonists",
            "abstract_snippet": "Review without hard endpoints.",
            "retrieval_tier": 1,
        },
        {
            "pmid": "8",
            "title": "SUSTAIN-6: cardiovascular outcomes trial",
            "abstract_snippet": "Major adverse cardiovascular events in T2DM.",
            "retrieval_tier": 2,
        },
    ]
    out = rerank_article_dicts(arts, _HERO, cap=2, clinical_intent=intent)
    assert out[0]["pmid"] == "8"
    assert "final_fusion" in (out[0].get("rank_score_debug") or {})
