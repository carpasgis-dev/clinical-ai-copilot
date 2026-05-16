"""
Forma semántica de la pregunta clínica para embedding / cross-encoder.

El reranker rankea contra intent estructurado (PICO-lite en inglés clínico),
no solo contra el texto crudo del usuario en español.
"""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_intent import (
    ClinicalIntent,
    intent_asks_cv_outcomes,
    intent_asks_renal_outcomes,
    intent_asks_safety_outcomes,
)
from app.capabilities.evidence_rag.outcome_ontology import semantic_outcome_phrases_en

_POP_EN: dict[str, str] = {
    "type 2 diabetes": "type 2 diabetes mellitus",
    "type 1 diabetes": "type 1 diabetes mellitus",
    "hypertension": "hypertension",
    "older adults": "older adults aged 65 years or older",
    "obesity": "obesity",
    "pcos": "polycystic ovary syndrome",
}

_IV_EN: dict[str, str] = {
    "sglt2": "SGLT2 inhibitors",
    "glp1": "GLP-1 receptor agonists",
    "metformin": "metformin",
    "insulin": "insulin",
    "anticoagulation": "anticoagulation",
}

_COMP_EN: dict[str, str] = {
    "metformin": "metformin monotherapy",
    "placebo": "placebo",
    "warfarin": "warfarin",
}

_OUTCOME_CV_EN = (
    "major adverse cardiovascular events (MACE), cardiovascular death, "
    "heart failure hospitalization, stroke, and myocardial infarction"
)
_OUTCOME_SAFETY_EN = "adverse events, hypoglycemia, tolerability, and treatment safety"
_OUTCOME_RENAL_EN = "chronic kidney disease, eGFR decline, albuminuria, and renal outcomes"
_OUTCOME_GLYCEMIC_EN = "HbA1c and glycemic control"

_PREF_EN: dict[str, str] = {
    "rct": "randomized controlled trials",
    "meta-analysis": "systematic reviews and meta-analyses",
    "guideline": "clinical practice guidelines",
    "cvot": "cardiovascular outcome trials (CVOT)",
}


def preferred_study_types_for_intent(intent: ClinicalIntent) -> list[str]:
    """Tipos de estudio preferidos derivados de intent + desenlaces."""
    prefs: list[str] = []
    for p in intent.evidence_preference:
        key = p.lower().replace(" ", "-")
        if key in _PREF_EN and key not in prefs:
            prefs.append(key)
    if intent_asks_cv_outcomes(intent) and "cvot" not in prefs:
        prefs.insert(0, "cvot")
    if "rct" not in prefs and intent.interventions:
        prefs.append("rct")
    if "meta-analysis" not in prefs:
        prefs.append("meta-analysis")
    return prefs


def build_intent_semantic_query(
    intent: ClinicalIntent | None,
    fallback_user_query: str,
) -> str:
    """
    Pregunta clínica en inglés para bi-encoder / cross-encoder.

    Ejemplo:
    «In older adults with type 2 diabetes and hypertension, do SGLT2 inhibitors or
    GLP-1 receptor agonists compared with metformin reduce cardiovascular outcomes …?»
    """
    if intent is None or not (
        intent.population or intent.interventions or intent.outcomes
    ):
        return (fallback_user_query or "").strip()

    pop_phrases = [_POP_EN.get(p.lower(), p) for p in intent.population[:6]]
    iv_phrases = [_IV_EN.get(i.lower(), i) for i in intent.interventions[:6]]
    comp_phrases = [_COMP_EN.get(c.lower(), c) for c in intent.comparator[:4]]

    pop_txt = ", ".join(pop_phrases) if pop_phrases else "patients"
    if intent.age_min is not None and intent.age_min >= 65:
        if "older adults" not in " ".join(pop_phrases).lower():
            pop_txt = f"older adults (≥{intent.age_min} years) with {pop_txt}"

    parts: list[str] = [f"In {pop_txt}"]

    if iv_phrases:
        iv_join = " or ".join(iv_phrases)
        if comp_phrases:
            comp_join = " versus ".join(comp_phrases)
            parts.append(
                f"what is the evidence that {iv_join} compared with {comp_join}"
            )
        else:
            parts.append(f"what is the evidence on {iv_join}")

    if intent_asks_cv_outcomes(intent):
        parts.append(f"for reducing {semantic_outcome_phrases_en(intent)}")
    elif intent_asks_safety_outcomes(intent):
        parts.append(f"regarding {_OUTCOME_SAFETY_EN}")
    elif intent_asks_renal_outcomes(intent):
        parts.append(f"regarding {_OUTCOME_RENAL_EN}")
    elif any(o.lower() == "glycemic control" for o in intent.outcomes):
        parts.append(f"regarding {_OUTCOME_GLYCEMIC_EN}")
    elif intent.outcomes:
        parts.append("for clinical outcomes: " + ", ".join(intent.outcomes[:6]))

    prefs = preferred_study_types_for_intent(intent)
    if prefs:
        pref_txt = ", ".join(_PREF_EN.get(p, p) for p in prefs[:5])
        parts.append(f"Prioritize evidence from {pref_txt}.")

    if intent_asks_cv_outcomes(intent):
        parts.append(
            "Exclude mechanistic pathophysiology reviews without reported clinical "
            "cardiovascular outcomes or randomized trial results."
        )

    sentence = " ".join(parts).strip()
    if not sentence.endswith("?"):
        sentence += "?"
    return sentence
