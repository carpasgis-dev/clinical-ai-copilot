"""Calibración de síntesis, tier-weight y bloques local/evidencia."""
from __future__ import annotations

from app.capabilities.evidence_rag.retrieval_tiers import (
    RetrievalTier,
    tier_retrieval_provenance_line,
    tier_weight_multiplier,
)
from app.capabilities.evidence_rag.evidence_rerank import rerank_article_dicts
from app.orchestration.medical_answer_builder import build_stub_medical_answer, render_medical_answer_to_text
from app.orchestration.synthesis_calibration import (
    calculate_synthesis_calibration,
    tier_aware_evidence_leadin,
)
from app.orchestration.llm_synthesis import (
    _inject_tier_provenance_in_pmid_sections,
    _sanitize_overconfident_synthesis,
    apply_tier_aware_evidence_summary,
)
from app.schemas.copilot_state import Route


def test_tier_weight_multiplier_ordering() -> None:
    assert tier_weight_multiplier(int(RetrievalTier.T1_EXACT_PICO)) > tier_weight_multiplier(4)
    assert tier_weight_multiplier(99) == 0.65


def test_tier_provenance_line_only_above_t1() -> None:
    assert tier_retrieval_provenance_line(1) is None
    assert tier_retrieval_provenance_line(2, "cvot_landmark") is not None
    assert "tier 4" in (tier_retrieval_provenance_line(4) or "")


def test_calculate_calibration_partial_primary_miss() -> None:
    state = {
        "evidence_bundle": {
            "retrieval_debug": {"outcome": "partial_primary_miss"},
            "articles": [
                {
                    "pmid": "1",
                    "title": "EMPA-REG OUTCOME trial cardiovascular",
                    "abstract_snippet": "MACE cardiovascular death heart failure",
                    "retrieval_tier": 2,
                    "retrieval_stage": "cvot_landmark",
                    "alignment_scores": {"outcome": 0.8},
                    "applicability_score": 0.7,
                },
            ],
        },
    }
    cal = calculate_synthesis_calibration(state)
    assert cal.retrieval_outcome == "partial_primary_miss"
    assert cal.dominant_retrieval_tier == 2
    assert cal.retrieval_confidence == 0.6


def test_tier_aware_leadin_high_tier() -> None:
    from app.orchestration.synthesis_calibration import SynthesisCalibration

    cal = SynthesisCalibration(
        retrieval_outcome="partial_primary_miss",
        dominant_retrieval_tier=4,
        retrieval_confidence=0.35,
    )
    lead = tier_aware_evidence_leadin(cal)
    assert lead is not None
    assert "expansión" in lead.lower() or "estricta" in lead.lower()


def test_apply_tier_aware_evidence_summary_prefix() -> None:
    state = {
        "synthesis_calibration": {
            "retrieval_outcome": "partial_primary_miss",
            "dominant_retrieval_tier": 4,
            "retrieval_confidence": 0.35,
            "evidence_specificity": 0.2,
            "applicability_confidence": 0.5,
            "primary_stage_hit": False,
            "landmark_present": False,
        },
    }
    ma = {"evidence_summary": "Se recuperaron 3 referencias."}
    out = apply_tier_aware_evidence_summary(ma, state)
    assert "expansión" in (out.get("evidence_summary") or "").lower()


def test_hybrid_medical_answer_separates_local_and_external() -> None:
    state = {
        "route": Route.HYBRID,
        "user_query": "diabetes e HTA",
        "sql_result": {"rows": [{"cohort_size": 1}]},
        "evidence_bundle": {
            "articles": [{"pmid": "99", "title": "SGLT2 and MACE", "abstract_snippet": "cardiovascular"}],
            "pmids": ["99"],
        },
    }
    ma = build_stub_medical_answer(state)
    assert ma.get("local_cohort_block")
    assert ma.get("external_evidence_block")
    assert "1 paciente" in ma["local_cohort_block"]
    rendered = render_medical_answer_to_text(ma)
    assert "## Cohorte local" in rendered
    assert "## Evidencia PubMed" in rendered
    assert "híbrida" in (ma.get("summary") or "").lower()


def test_rerank_applies_tier_weight() -> None:
    arts = [
        {
            "pmid": "1",
            "title": "Randomized trial SGLT2 cardiovascular outcomes MACE",
            "abstract_snippet": "cardiovascular death heart failure MACE",
            "retrieval_tier": 4,
        },
        {
            "pmid": "2",
            "title": "EMPA-REG OUTCOME randomized cardiovascular",
            "abstract_snippet": "MACE cardiovascular mortality",
            "retrieval_tier": 2,
        },
    ]
    ranked = rerank_article_dicts(
        arts,
        "SGLT2 diabetes cardiovascular events",
        cap=2,
    )
    assert ranked
    t2 = next(a for a in ranked if a.get("pmid") == "2")
    assert t2.get("rank_score_debug", {}).get("tier_weight") == tier_weight_multiplier(2)


def test_sanitize_overconfident_when_low_confidence() -> None:
    from app.orchestration.synthesis_calibration import SynthesisCalibration

    cal = SynthesisCalibration(retrieval_confidence=0.3, dominant_retrieval_tier=4)
    text = "El estudio demuestra reducción significativa de eventos cardiovasculares."
    out = _sanitize_overconfident_synthesis(text, cal)
    assert "Nota:" in out or "prudente" in out.lower()


def test_inject_tier_provenance_in_pmid_sections() -> None:
    facts = {
        "citas_pubmed": [
            {
                "pmid": "123",
                "nota_procedencia_recuperacion": "*Recuperado vía expansión (tier 4).*",
            }
        ]
    }
    import json

    text = "### PMID 123 — Trial X\nContenido del abstract.\n"
    out = _inject_tier_provenance_in_pmid_sections(text, json.dumps(facts))
    assert "*Recuperado vía expansión" in out
