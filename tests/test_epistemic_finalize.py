"""Epistémica post-fusión, rerank y score en reasoning."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_intent import extract_clinical_intent
from app.capabilities.evidence_rag.epistemic_ranking import (
    finalize_rank_score,
    infer_epistemic_profile,
)
from app.capabilities.evidence_rag.evidence_rerank import rerank_article_dicts
from app.capabilities.evidence_rag.intent_semantic_query import build_intent_semantic_query
from app.orchestration.reasoning import build_reasoning_state
from app.schemas.copilot_state import ClinicalContext

_HERO = (
    "En pacientes con diabetes e hipertensión ≥65 años, "
    "¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 "
    "frente a solo metformina para reducir eventos cardiovasculares?"
)


def test_potential_mechanisms_detected_as_mechanistic() -> None:
    intent = extract_clinical_intent(_HERO)
    ep = infer_epistemic_profile(
        "Potential mechanisms of cardiac and renal fibrosis with SGLT2",
        "pathophysiology and fibrosis pathways",
        intent=intent,
    )
    assert ep.evidence_type == "mechanistic"
    assert ep.multiplier == 0.15


def test_finalize_rank_score_crushes_mechanistic_high_semantic() -> None:
    intent = extract_clinical_intent(_HERO)
    final, meta = finalize_rank_score(
        0.95,
        title="Potential mechanisms of cardiac fibrosis and SGLT2 inhibitors",
        abstract="mechanisms of renal fibrosis",
        clinical_intent=intent,
    )
    assert meta["evidence_type"] == "mechanistic"
    assert final < 0.2
    assert meta["fused"] == round(final, 4)


def test_rerank_mechanistic_below_cvot_when_semantic_off() -> None:
    intent = extract_clinical_intent(_HERO)
    arts = [
        {
            "pmid": "999",
            "title": "Potential mechanisms of cardiac and renal fibrosis with empagliflozin",
            "abstract_snippet": "pathophysiology fibrosis mechanisms SGLT2",
        },
        {
            "pmid": "1",
            "title": "EMPA-REG OUTCOME trial cardiovascular death heart failure randomized",
            "abstract_snippet": "MACE cardiovascular death randomized controlled trial",
        },
    ]
    ranked = rerank_article_dicts(arts, _HERO, cap=2, clinical_intent=intent)
    assert ranked[0]["pmid"] == "1"


def test_intent_semantic_query_excludes_mechanistic_reviews() -> None:
    ctx = ClinicalContext(population_conditions=["diabet", "hipertens"], population_age_min=65)
    intent = extract_clinical_intent(_HERO, ctx)
    q = build_intent_semantic_query(intent, _HERO).lower()
    assert "exclude" in q
    assert "mechanistic" in q


def test_reasoning_uses_semantic_scores_fused() -> None:
    state = {
        "route": "evidence",
        "user_query": _HERO,
        "evidence_bundle": {
            "articles": [
                {
                    "pmid": "42",
                    "title": "Trial",
                    "abstract_snippet": "MACE",
                    "semantic_scores": {"fused": 0.123},
                },
            ],
        },
    }
    rs = build_reasoning_state(state)
    assert rs.evidence_assessments[0].relevance_score == 0.123
