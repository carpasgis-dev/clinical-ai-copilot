from __future__ import annotations
import re
from dataclasses import dataclass
from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent, intent_asks_cv_outcomes
from app.capabilities.clinical_sql.terminology import fold_ascii

# Evidence Intent Classification
EVIDENCE_TYPES = {
    "mechanistic",
    "preclinical",
    "observational",
    "target_trial",
    "rct",
    "meta_analysis",
    "guideline",
    "review",
    "case_report"
}

_EVIDENCE_PATTERNS = [
    (re.compile(r"\bmeta-analysis\b|\bmeta analysis\b|\bsystematic review\b|\bpooled analysis\b", re.I), "meta_analysis"),
    (re.compile(r"\bguideline\b|\brecommendation[s]?\b", re.I), "guideline"),
    (re.compile(r"\brandomized\s+controlled\b|\brct\b|\bdouble-blind\b|\bplacebo-controlled\b|\bphase 3\b|\bphase iii\b|\bmulticenter\b|\bpragmatic trial\b|\bemulation\b", re.I), "rct"),
    (re.compile(r"\btarget\s+trial\b", re.I), "target_trial"),
    (re.compile(r"\bobservational\b|\bcohort\s+study\b|\bretrospective\b|\bprospective\b|\breal-world\b", re.I), "observational"),
    (re.compile(r"\bnarrative\s+review\b|\breview\b", re.I), "review"),
    (re.compile(r"\bmolecular\s+mechanism\b|\bmechanistic\s+pathway\b|\bpathogenesis\b|\bpotential\s+mechanisms\b|\bmechanisms\s+of\b|\bpathophysiology\b|\bmechanisms\b", re.I), "mechanistic"),
    (re.compile(r"\banimal\s+model\b|\bmurine\b|\bmice\b|\brats\b|\bzebrafish\b|\bin\s+vitro\b|\bcell\s+line\b", re.I), "preclinical"),
    (re.compile(r"\bcase\s+report\b|\bcase\s+series\b", re.I), "case_report")
]

@dataclass
class EpistemicProfile:
    evidence_type: str
    multiplier: float
    boost: float

def infer_epistemic_profile(title: str, abstract: str, intent: ClinicalIntent | None = None) -> EpistemicProfile:
    blob = fold_ascii(f"{title} {abstract}")
    detected = "unknown"
    for pat, ev_type in _EVIDENCE_PATTERNS:
        if pat.search(blob):
            detected = ev_type
            break
            
    multiplier = 1.0
    boost = 0.0
    
    # Defaults
    if detected in ("preclinical", "mechanistic"):
        multiplier = 0.25 # heavy penalty
    elif detected == "case_report":
        multiplier = 0.45
    elif detected in ("rct", "meta_analysis", "guideline", "target_trial"):
        boost = 0.15
    elif detected == "review":
        multiplier = 0.55

    # If the user is specifically asking for outcomes/treatment
    if intent and (intent.outcomes or intent_asks_cv_outcomes(intent)):
        if detected in ("preclinical", "mechanistic"):
            multiplier = 0.15 # even heavier penalty for outcomes
        elif detected in ("rct", "meta_analysis", "guideline"):
            boost = 0.30 # heavy boost for outcomes
        elif detected == "review":
            multiplier = 0.32

    return EpistemicProfile(
        evidence_type=detected,
        multiplier=multiplier,
        boost=boost
    )


def finalize_rank_score(
    base_score: float,
    *,
    title: str,
    abstract: str = "",
    clinical_intent: ClinicalIntent | None = None,
) -> tuple[float, dict[str, float | str]]:
    """
    Capa única post-fusión / post-heurística: epistémica × supresión de ruido estructural.

    ``fused`` en el dict devuelto es el score final para ordenar y para ``reasoning``.
    """
    from app.capabilities.evidence_rag.noise_suppression import apply_noise_suppression

    ep = infer_epistemic_profile(title, abstract, intent=clinical_intent)
    after_epistemic = base_score * ep.multiplier * (1.0 + ep.boost)
    final, ns = apply_noise_suppression(
        after_epistemic,
        title,
        abstract,
        clinical_intent=clinical_intent,
    )
    return max(0.01, final), {
        "fused_pre_epistemic": round(base_score, 4),
        "epistemic_multiplier": ep.multiplier,
        "epistemic_boost": round(ep.boost, 4),
        "evidence_type": ep.evidence_type,
        "noise_multiplier": ns.multiplier,
        "fused": round(final, 4),
    }
