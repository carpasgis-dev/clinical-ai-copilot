import re
from dataclasses import dataclass
from typing import Optional, List

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.schemas.copilot_state import ClinicalContext


@dataclass
class ApplicabilityProfile:
    applicability_score: float
    explanation: Optional[str]
    demographic_match: bool


def calculate_applicability(
    title: str,
    abstract: str,
    clinical_context: Optional[ClinicalContext]
) -> ApplicabilityProfile:
    """
    Evaluates patient-to-paper applicability grounding.
    Uses age limits, cohort metadata (conditions, etc) to penalize or boost
    the fetched article based on population alignment.
    """
    if not clinical_context:
        return ApplicabilityProfile(applicability_score=1.0, explanation=None, demographic_match=False)

    blob = fold_ascii(f"{title} {abstract}")
    score_mult = 1.0
    explanations: List[str] = []
    
    # 1. Demographics & Age Range penalties
    age_str = str(clinical_context.age_range or "")
    cohort_is_old = ">65" in age_str or "65" in age_str or "70" in age_str or "80" in age_str
    
    pediatric_regex = re.compile(r"\b(adolescents?|pediatric|paediatric|children|childhood|infants?|teens?|teenage|youth|neonatal)\b", re.I)
    pregnancy_regex = re.compile(r"\b(pregnan\w*|gestation\w*|prenatal|obstetric|trimester|maternal|gravida)\b", re.I)
    old_regex = re.compile(r"\b(older adult|elderly|aged\s+65|≥\s*65|65\s*\+|geriatric)\b", re.I)
    
    has_pediatric = pediatric_regex.search(blob)
    has_pregnancy = pregnancy_regex.search(blob)
    has_old = old_regex.search(blob)

    if cohort_is_old:
        if has_pediatric:
            score_mult *= 0.5  # Heavy penalty for pediatric data applied to seniors
            explanations.append("Limitada: población pediátrica frente a cohorte ≥65 años.")
        if has_pregnancy:
            score_mult *= 0.5
            explanations.append("Limitada: cohorte gestacional frente a senil.")
        if has_old:
            score_mult *= 1.25 # Boost!
            explanations.append("Alineación: adulto mayor mencionado explicitamente.")
            
    # Penalize if context has no pregnancy signals, but article heavily features it
    # We check if context.conditions mentions pregnancy
    ctx_blob = fold_ascii(" ".join((clinical_context.conditions or []) + (clinical_context.population_conditions or [])))
    ctx_has_pregnancy = pregnancy_regex.search(ctx_blob)
    
    if has_pregnancy and not ctx_has_pregnancy:
        score_mult *= 0.7  # Penalty
        explanations.append("Limitada: artículo de cohorte obstétrica no justificada por el paciente.")

    # 2. Main conditions profile match
    conditions = set(clinical_context.population_conditions or []) | set(clinical_context.conditions or [])
    matched_conditions = 0
    for c in conditions:
        if len(c) > 3 and fold_ascii(c) in blob:
            matched_conditions += 1
            
    if matched_conditions > 0 and len(conditions) > 0:
        boost = 1.0 + (0.10 * (matched_conditions / len(conditions)))
        score_mult *= boost
        explanations.append(f"Alineación positiva: {matched_conditions} comorbilidades concurrentes detectadas.")

    explanation_str = " | ".join(explanations) if explanations else None
    demographic_match = len(explanations) > 0 and score_mult > 1.0

    return ApplicabilityProfile(
        applicability_score=round(score_mult, 4),
        explanation=explanation_str,
        demographic_match=demographic_match
    )
