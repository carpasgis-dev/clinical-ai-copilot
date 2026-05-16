"""
Capa fina sobre ``landmark_registry``: boost de rerank e hints de síntesis.

Los datos de trials y cláusulas PubMed viven en ``landmark_registry``.
"""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent, intent_asks_cv_outcomes
from app.capabilities.evidence_rag.landmark_registry import (
    LANDMARK_ANTICOAG_TRIALS,
    LANDMARK_CVOTS,
    LandmarkTrial,
    landmark_anticoag_retrieval_clause,
    landmark_cvot_retrieval_clause,
    landmark_pubmed_acronyms_clause,
    landmark_pubmed_drugs_clause,
    match_anticoag_landmark,
    match_diabetes_cvot_landmark,
    match_landmark_trial,
    trials_for_drug_classes,
)

__all__ = [
    "LANDMARK_ANTICOAG_TRIALS",
    "LANDMARK_CVOTS",
    "LandmarkTrial",
    "landmark_anticoag_retrieval_clause",
    "landmark_cvot_retrieval_clause",
    "landmark_pubmed_acronyms_clause",
    "landmark_pubmed_drugs_clause",
    "landmark_rerank_boost",
    "landmark_synthesis_hint",
    "match_anticoag_landmark",
    "match_diabetes_cvot_landmark",
    "match_landmark_trial",
    "trials_for_drug_classes",
]


def landmark_rerank_boost(
    title: str,
    abstract_snippet: str = "",
    *,
    clinical_intent: ClinicalIntent | None = None,
) -> float:
    if clinical_intent is not None and not intent_asks_cv_outcomes(clinical_intent):
        return 0.0
    trial = match_diabetes_cvot_landmark(title, abstract_snippet) or match_landmark_trial(
        title, abstract_snippet
    )
    if not trial:
        return 0.0
    if trial.evidence_level == "rct_landmark":
        return 0.28
    return 0.14


def landmark_synthesis_hint(title: str, abstract_snippet: str = "") -> str | None:
    trial = match_landmark_trial(title, abstract_snippet)
    if not trial:
        return None
    drugs = ", ".join(trial.drugs)
    outs = ", ".join(trial.outcomes[:3])
    return (
        f"Ensayo landmark conocido ({trial.acronym}): {drugs}; "
        f"población {trial.population}; desenlaces {outs}."
    )
