"""Answerability factorizado, roles inferenciales y breakdown."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_answerability import (
    EVIDENCE_ROLE_PRIMARY,
    EVIDENCE_ROLE_SECONDARY,
    compute_answerability_breakdown,
    eligible_for_featured_headline,
    infer_evidence_role,
    passes_answerability_gate,
)
from app.capabilities.evidence_rag.evidence_rerank import infer_study_type_from_title


def test_post_hoc_landmark_is_secondary_not_primary() -> None:
    tit = "Post hoc analysis of SUSTAIN-6: semaglutide and cardiovascular outcomes"
    snip = "Subgroup analysis from the SUSTAIN-6 trial."
    st = infer_study_type_from_title(tit)
    role = infer_evidence_role(st, tit, snip)
    assert role == EVIDENCE_ROLE_SECONDARY
    bd = compute_answerability_breakdown(st, tit, snip, question_type="treatment_efficacy")
    assert bd.final_score < 0.55
    assert bd.landmark_signal >= 0.3
    assert bd.secondary_analysis_penalty < 1.0


def test_primary_cvot_scores_above_post_hoc() -> None:
    primary_tit = "SUSTAIN-6: cardiovascular outcomes trial in type 2 diabetes"
    primary_snip = "Randomized controlled trial; primary endpoint major adverse cardiovascular events."
    post_tit = "Post hoc analysis of SUSTAIN-6 outcomes in elderly patients"
    post_snip = "Secondary analysis of the SUSTAIN-6 cohort."
    st_p = infer_study_type_from_title(primary_tit)
    st_post = infer_study_type_from_title(post_tit)
    score_primary = compute_answerability_breakdown(
        st_p, primary_tit, primary_snip, question_type="treatment_efficacy"
    ).final_score
    score_post = compute_answerability_breakdown(
        st_post, post_tit, post_snip, question_type="treatment_efficacy"
    ).final_score
    assert score_primary > score_post + 0.15
    assert infer_evidence_role(st_p, primary_tit, primary_snip) == EVIDENCE_ROLE_PRIMARY


def test_telemedicine_noise_fails_gate_for_comparative() -> None:
    tit = "ANMCO position paper on E-health integration in cardiology"
    snip = "Telemedicine and digital health for atrial fibrillation."
    assert not passes_answerability_gate(
        tit,
        snip,
        question_type="comparative_effectiveness",
    )


def test_basic_research_excluded_from_featured_when_strong_pool() -> None:
    tit = "Oxidative stress in experimental model of diabetes pathophysiology"
    snip = "In vitro murine oxidative stress pathway."
    st = infer_study_type_from_title(tit)
    assert not eligible_for_featured_headline(
        st,
        tit,
        snip,
        pool_has_strong_answerable=True,
    )


def test_breakdown_exposes_components() -> None:
    tit = "EMPA-REG OUTCOME: cardiovascular outcomes with empagliflozin"
    snip = "Randomized trial; MACE and cardiovascular death in type 2 diabetes."
    st = infer_study_type_from_title(tit)
    bd = compute_answerability_breakdown(st, tit, snip, question_type="treatment_efficacy")
    d = bd.to_dict()
    assert "intervention_alignment" in d
    assert "landmark_signal" in d
    assert bd.evidence_role == EVIDENCE_ROLE_PRIMARY
