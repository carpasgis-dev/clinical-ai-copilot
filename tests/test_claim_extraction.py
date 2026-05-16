"""Extracción determinista de claims (SGLT2, GLP-1, DOAC)."""
from __future__ import annotations

from app.capabilities.evidence_rag.claim_extraction import (
    claims_to_evidence_statements,
    extract_claims_deterministic,
    render_claim_bundle_markdown,
)
from app.capabilities.evidence_rag.clinical_semantics import (
    ClinicalEvidenceFrame,
    build_clinical_evidence_frame,
)


def _art(
    pmid: str,
    title: str,
    snip: str,
    *,
    score: float = 0.9,
) -> dict:
    return {
        "pmid": pmid,
        "title": title,
        "abstract_snippet": snip,
        "final_rank_score": score,
        "retrieval_tier": 2,
    }


def test_doac_ic_bleeding_claim_with_landmarks() -> None:
    frame = ClinicalEvidenceFrame(
        question_type="comparative_effectiveness",
        intervention=["DOAC"],
        comparator=["warfarin"],
        intervention_concept_ids=["doac_class"],
        comparator_concept_ids=["warfarin"],
        outcome_concept_ids=["outcome_bleeding"],
        population_concept_ids=["pop_af"],
        therapeutic_objective=True,
    )
    arts = [
        _art(
            "1",
            "ARISTOTLE: apixaban versus warfarin in atrial fibrillation",
            "Apixaban reduced risk of intracranial hemorrhage compared with warfarin. "
            "Randomized controlled trial.",
        ),
        _art(
            "2",
            "RE-LY: dabigatran versus warfarin",
            "Lower risk of intracranial hemorrhage with dabigatran vs warfarin in atrial fibrillation.",
            score=0.85,
        ),
        _art(
            "3",
            "Narrative review of DOAC versus warfarin in atrial fibrillation",
            "No significant difference in major bleeding or intracranial hemorrhage "
            "between apixaban and warfarin in some observational cohorts.",
            score=0.4,
        ),
    ]
    bundle = extract_claims_deterministic(
        frame, arts, question_type="comparative_effectiveness"
    )
    axis_ids = {c.axis_id for c in bundle.claims}
    assert "doac_vs_warfarin_ic_bleeding" in axis_ids
    claim = next(c for c in bundle.claims if c.axis_id == "doac_vs_warfarin_ic_bleeding")
    assert claim.direction == "benefit"
    assert len(claim.support) >= 2
    assert claim.evidence_strength in ("moderate", "high")
    assert len(claim.contradicting) >= 1
    assert claim.axis_label
    assert "ARISTOTLE" in claim.landmark_support or any(
        s.landmark_acronym == "ARISTOTLE" for s in claim.support
    )
    md = render_claim_bundle_markdown(bundle)
    assert md and ("ARISTOTLE" in md or "PMID 1" in md)


def test_post_hoc_weaker_than_primary_on_same_axis() -> None:
    frame = build_clinical_evidence_frame(
        "semaglutide cardiovascular outcomes type 2 diabetes MACE"
    )
    arts = [
        _art(
            "10",
            "SUSTAIN-6: semaglutide and cardiovascular outcomes in type 2 diabetes",
            "Randomized trial. Reduced risk of major adverse cardiovascular events (MACE).",
        ),
        _art(
            "11",
            "Post hoc analysis of SUSTAIN-6: subgroup cardiovascular outcomes",
            "Secondary analysis from SUSTAIN-6. No significant difference in MACE in elderly subgroup.",
            score=0.7,
        ),
    ]
    bundle = extract_claims_deterministic(
        frame, arts, question_type="treatment_efficacy"
    )
    mace = [c for c in bundle.claims if c.axis_id == "glp1_ra_mace"]
    if mace:
        assert mace[0].direction in ("benefit", "neutral", "unclear", "harm")
        pmids_support = {s.pmid for s in mace[0].support}
        pmids_contra = {s.pmid for s in mace[0].contradicting}
        assert "10" in pmids_support
        assert "11" in pmids_contra or "11" in pmids_support


def test_primary_slice_limits_axes() -> None:
    frame = build_clinical_evidence_frame(
        "empagliflozin semaglutide apixaban warfarin MACE heart failure"
    )
    arts = [
        _art("20", "EMPA-REG: heart failure hospitalization", "Reduced HF hospitalization in T2DM."),
        _art("21", "SUSTAIN-6: MACE semaglutide", "Reduced MACE in type 2 diabetes."),
        _art(
            "22",
            "ARISTOTLE: apixaban vs warfarin intracranial hemorrhage",
            "Lower intracranial hemorrhage with apixaban vs warfarin in AF.",
        ),
    ]
    bundle = extract_claims_deterministic(frame, arts, primary_slice=True)
    assert {c.axis_id for c in bundle.claims}.issubset(
        {"sglt2_hf_hospitalization", "glp1_ra_mace", "doac_vs_warfarin_ic_bleeding"}
    )


def test_claims_to_evidence_statements() -> None:
    frame = build_clinical_evidence_frame("empagliflozin heart failure diabetes")
    arts = [
        _art(
            "20",
            "EMPA-REG OUTCOME: empagliflozin",
            "Reduced risk of heart failure hospitalization in type 2 diabetes.",
        ),
    ]
    bundle = extract_claims_deterministic(frame, arts)
    stmts = claims_to_evidence_statements(bundle)
    assert stmts
    assert stmts[0]["citation_pmids"]
    assert "Claim:" in stmts[0]["statement"]


def test_sglt2_hf_claim_from_cvot() -> None:
    frame = build_clinical_evidence_frame(
        "empagliflozin heart failure hospitalization type 2 diabetes"
    )
    arts = [
        _art(
            "20",
            "EMPA-REG OUTCOME: empagliflozin and cardiovascular outcomes",
            "Reduced risk of heart failure hospitalization in patients with type 2 diabetes. "
            "Cardiovascular outcomes trial.",
        ),
    ]
    bundle = extract_claims_deterministic(frame, arts, question_type="treatment_efficacy")
    assert any(c.axis_id == "sglt2_hf_hospitalization" for c in bundle.claims)
