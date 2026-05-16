"""
Calcula un objeto de calibración de síntesis para guiar la generación de respuestas y la interfaz de usuario.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.capabilities.evidence_rag.clinical_knowledge import (
    landmark_synthesis_hint,
    match_diabetes_cvot_landmark,
)
from app.capabilities.evidence_rag.retrieval_tiers import RetrievalTier
from app.orchestration.evidence_dedup import deduplicate_citations


@dataclass
class SynthesisCalibration:
    """
    Señales de confianza sobre la evidencia recuperada, para síntesis tier-aware.
    """

    retrieval_outcome: str = "pending"  # success | partial_primary_miss | full_miss | zero_hits_all_stages
    dominant_retrieval_tier: int = 99
    retrieval_confidence: float = 0.0
    evidence_specificity: float = 0.0
    applicability_confidence: float = 0.0
    primary_stage_hit: bool = False
    landmark_present: bool = False
    raw_top_evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw_top_evidence", None)
        return d


def tier_aware_evidence_leadin(cal: SynthesisCalibration) -> str | None:
    """
    Párrafo determinista al inicio de evidence_summary cuando la recuperación es débil o amplia.
    """
    if cal.dominant_retrieval_tier < 3 and cal.retrieval_confidence >= 0.5:
        return None
    parts: list[str] = []
    if cal.retrieval_outcome == "partial_primary_miss":
        parts.append(
            "La búsqueda PubMed estricta (etapa PICO primaria) no recuperó candidatos suficientes; "
            "parte de la evidencia procede de etapas ampliadas (landmark, revisión o expansión semántica)."
        )
    if cal.dominant_retrieval_tier >= 4:
        parts.append(
            "La evidencia mostrada proviene mayoritariamente de expansión semántica o contexto amplio "
            "(tier ≥4); la aplicabilidad directa a la población e intervención preguntadas puede ser limitada."
        )
    elif cal.dominant_retrieval_tier == 3:
        parts.append(
            "Predominan revisiones o síntesis (tier 3); pueden no reflejar RCT directos en la cohorte exacta."
        )
    if cal.retrieval_confidence < 0.5:
        parts.append(
            "Confianza de recuperación limitada (heurística del sistema); contrastar con búsqueda manual en PubMed."
        )
    return " ".join(parts) if parts else None


def calculate_synthesis_calibration(state: dict[str, Any]) -> SynthesisCalibration:
    """Calibración tras retrieval + rerank (estado del grafo)."""
    evidence_bundle = state.get("evidence_bundle") or {}
    articles = evidence_bundle.get("articles") or []

    top_articles = deduplicate_citations([a for a in articles if isinstance(a, dict)])
    top_n = 6
    top_articles = top_articles[:top_n]

    cal = SynthesisCalibration(raw_top_evidence=top_articles)

    rd = evidence_bundle.get("retrieval_debug") if isinstance(evidence_bundle, dict) else None
    if isinstance(rd, dict) and rd.get("outcome"):
        cal.retrieval_outcome = str(rd["outcome"])

    if not top_articles:
        cal.retrieval_outcome = cal.retrieval_outcome if cal.retrieval_outcome != "pending" else "full_miss"
        cal.retrieval_confidence = 0.0
        return cal

    tiers = [int(a.get("retrieval_tier", 99)) for a in top_articles]
    cal.dominant_retrieval_tier = max(tiers) if tiers else 99

    cal.primary_stage_hit = any(t <= int(RetrievalTier.T1_EXACT_PICO) for t in tiers)

    if cal.retrieval_outcome == "pending":
        if cal.primary_stage_hit:
            cal.retrieval_outcome = "success"
        else:
            cal.retrieval_outcome = "partial_primary_miss"

    if cal.retrieval_outcome == "success" and cal.dominant_retrieval_tier <= 2:
        cal.retrieval_confidence = 1.0
    elif cal.retrieval_outcome == "partial_primary_miss" and cal.dominant_retrieval_tier <= 2:
        cal.retrieval_confidence = 0.6
    elif cal.dominant_retrieval_tier >= 4:
        cal.retrieval_confidence = 0.35
    else:
        cal.retrieval_confidence = 0.5

    cal.landmark_present = any(
        match_diabetes_cvot_landmark(a.get("title", ""), a.get("abstract_snippet", ""))
        for a in top_articles
    )

    alignment_scores = [a.get("alignment_scores", {}) for a in top_articles]
    applicability_scores = [float(a.get("applicability_score") or 0.0) for a in top_articles]

    high_outcome_alignment = sum(
        1 for scores in alignment_scores
        if isinstance(scores, dict) and float(scores.get("outcome", 0.0) or 0.0) > 0.7
    )
    landmark_n = sum(
        1 for a in top_articles
        if landmark_synthesis_hint(a.get("title", ""), a.get("abstract_snippet", ""))
    )
    num_specific = high_outcome_alignment + landmark_n
    cal.evidence_specificity = min(1.0, num_specific / len(top_articles)) if top_articles else 0.0

    cal.applicability_confidence = (
        sum(applicability_scores) / len(applicability_scores) if applicability_scores else 0.0
    )

    return cal


def calibration_from_state(state: dict[str, Any]) -> SynthesisCalibration | None:
    raw = state.get("synthesis_calibration")
    if raw is None:
        return None
    if isinstance(raw, SynthesisCalibration):
        return raw
    if isinstance(raw, dict):
        return SynthesisCalibration(
            retrieval_outcome=str(raw.get("retrieval_outcome") or "pending"),
            dominant_retrieval_tier=int(raw.get("dominant_retrieval_tier") or 99),
            retrieval_confidence=float(raw.get("retrieval_confidence") or 0.0),
            evidence_specificity=float(raw.get("evidence_specificity") or 0.0),
            applicability_confidence=float(raw.get("applicability_confidence") or 0.0),
            primary_stage_hit=bool(raw.get("primary_stage_hit")),
            landmark_present=bool(raw.get("landmark_present")),
        )
    return None
