"""Intención clínica y alineación dinámica (PICO-lite)."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_alignment import (
    alignment_composite,
    score_paper_alignment,
)
from app.capabilities.evidence_rag.clinical_intent import extract_clinical_intent
from app.capabilities.evidence_rag.evidence_rerank import rerank_article_dicts
from app.schemas.copilot_state import ClinicalContext, Route
from app.orchestration.medical_answer_builder import build_stub_medical_answer


_HERO_QUERY = (
    "En pacientes con diabetes e hipertensión ≥65 años, "
    "¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 "
    "frente a solo metformina para reducir eventos cardiovasculares?"
)


def test_extract_clinical_intent_hero_query() -> None:
    intent = extract_clinical_intent(_HERO_QUERY)
    assert "type 2 diabetes" in intent.population
    assert "hypertension" in intent.population
    assert "older adults" in intent.population
    assert "sglt2" in intent.interventions
    assert "glp1" in intent.interventions
    assert "metformin" in intent.comparator or "metformin" in intent.interventions
    assert any(o in intent.outcomes for o in ("cardiovascular events", "mace", "heart failure"))


def test_pcos_paper_low_population_score() -> None:
    intent = extract_clinical_intent(_HERO_QUERY)
    scores = score_paper_alignment(
        intent,
        "PCOS and metabolic syndrome in adolescents: a systematic review",
        "polycystic ovary syndrome adolescents insulin resistance",
    )
    assert scores.population_score < 0.35
    assert alignment_composite(scores) < 0.45


def test_t2dm_cv_trial_high_alignment() -> None:
    intent = extract_clinical_intent(_HERO_QUERY)
    scores = score_paper_alignment(
        intent,
        "SGLT2 inhibitors and cardiovascular outcomes in older adults with type 2 diabetes and hypertension",
        "randomized trial MACE heart failure mortality empagliflozin",
    )
    assert scores.population_score >= 0.5
    assert scores.intervention_score >= 0.5
    assert scores.outcome_score >= 0.5
    assert alignment_composite(scores, priority_axis=intent.priority_axis, intent=intent) >= 0.55


def test_depression_glp1_low_outcome_vs_cv_trial() -> None:
    intent = extract_clinical_intent(_HERO_QUERY)
    dep = score_paper_alignment(
        intent,
        "Risk of Depression with GLP-1 Receptor Agonists in Type 2 Diabetes: A Meta-Analysis",
        "depressive symptoms quality of life semaglutide liraglutide type 2 diabetes",
    )
    cv = score_paper_alignment(
        intent,
        "SGLT2 inhibitors reduce heart failure hospitalization and MACE in type 2 diabetes",
        "cardiovascular death stroke hypertension older adults empagliflozin trial",
    )
    assert dep.outcome_score < cv.outcome_score
    assert dep.outcome_score < 0.35
    assert alignment_composite(dep, priority_axis=intent.priority_axis, intent=intent) < alignment_composite(
        cv, priority_axis=intent.priority_axis, intent=intent
    )


def test_rerank_demotes_depression_glp1_paper() -> None:
    intent = extract_clinical_intent(_HERO_QUERY)
    arts = [
        {
            "pmid": "dep",
            "title": "Risk of Depression with GLP-1 Receptor Agonists in Type 2 Diabetes",
            "abstract_snippet": "depressive symptoms meta-analysis semaglutide",
        },
        {
            "pmid": "cv",
            "title": "Empagliflozin and cardiovascular outcomes in type 2 diabetes",
            "abstract_snippet": "MACE heart failure hospitalization cardiovascular death",
        },
    ]
    out = rerank_article_dicts(
        arts,
        _HERO_QUERY,
        cap=2,
        population_age_min=65,
        clinical_intent=intent,
    )
    assert out[0]["pmid"] == "cv"


def test_glp1_weight_plus_mace_not_penalized() -> None:
    """Trial con reducción de peso Y MACE: no debe penalizarse por off-topic weight."""
    intent = extract_clinical_intent(_HERO_QUERY)
    scores = score_paper_alignment(
        intent,
        "Semaglutide and cardiovascular outcomes in type 2 diabetes with body mass reduction",
        "BMI decrease MACE heart failure hospitalization adiposity anthropometric outcomes",
    )
    assert scores.outcome_score >= 0.70


def test_cv_outcome_graded_tiers() -> None:
    from app.capabilities.evidence_rag.clinical_alignment import _cv_outcome_graded_score

    generic_only = _cv_outcome_graded_score("cardiovascular risk reduction in diabetes")
    mace_hf = _cv_outcome_graded_score("MACE and heart failure hospitalization cardiovascular death")
    assert 0.35 <= generic_only <= 0.50
    assert mace_hf >= 0.85


def test_comparator_metformin_beats_drug_class_only() -> None:
    intent = extract_clinical_intent(_HERO_QUERY)
    vs_met = score_paper_alignment(
        intent,
        "SGLT2 inhibitor versus metformin on cardiovascular outcomes in type 2 diabetes",
        "compared with metformin monotherapy MACE heart failure",
    )
    class_only = score_paper_alignment(
        intent,
        "Semaglutide versus liraglutide in patients with type 2 diabetes",
        "glycemic control HbA1c randomized trial",
    )
    assert vs_met.comparator_score > class_only.comparator_score
    assert vs_met.comparator_score >= 0.85
    assert class_only.comparator_score <= 0.3


def test_rerank_demotes_pcos_with_clinical_intent() -> None:
    intent = extract_clinical_intent(_HERO_QUERY)
    arts = [
        {
            "pmid": "1",
            "title": "PCOS and metabolic syndrome in adolescents: meta-analysis",
            "abstract_snippet": "adolescents polycystic ovary",
        },
        {
            "pmid": "2",
            "title": "SGLT2 inhibitors for cardiovascular risk in older adults with type 2 diabetes",
            "abstract_snippet": "hypertension MACE heart failure randomized",
        },
    ]
    out = rerank_article_dicts(
        arts,
        _HERO_QUERY,
        cap=2,
        population_age_min=65,
        population_conditions=["diabetes", "hipertensión"],
        clinical_intent=intent,
    )
    assert out[0]["pmid"] == "2"
    assert "alignment_scores" in out[0]


def test_build_stub_includes_clinical_intent_from_state() -> None:
    state = {
        "route": Route.HYBRID,
        "user_query": _HERO_QUERY,
        "clinical_intent": extract_clinical_intent(_HERO_QUERY).to_dict(),
        "evidence_bundle": {
            "articles": [
                {
                    "pmid": "99",
                    "title": "SGLT2 and MACE in type 2 diabetes",
                    "abstract_snippet": "older adults hypertension cardiovascular events",
                    "alignment_scores": {
                        "population_score": 0.9,
                        "intervention_score": 1.0,
                        "outcome_score": 0.85,
                    },
                }
            ],
            "pmids": ["99"],
        },
        "clinical_context": ClinicalContext(
            population_conditions=["diabet"],
            population_medications=["metform"],
            population_age_min=65,
        ).model_dump(mode="json"),
    }
    ma = build_stub_medical_answer(state)
    assert len(ma["citations"]) == 1
