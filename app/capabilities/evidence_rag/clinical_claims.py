"""
Capa de claims clínicos: afirmaciones agregadas con soporte y conflictos explícitos.

Extracción determinista en ``claim_extraction``; síntesis LLM redacta desde ``ClaimBundle``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EvidenceStrength = Literal["high", "moderate", "low", "insufficient"]
ClaimDirection = Literal["benefit", "harm", "neutral", "unclear"]
ClaimConsistency = Literal["consistent", "mixed", "conflicting", "insufficient"]


@dataclass
class ClaimSupport:
    pmid: str
    study_type: str | None = None
    evidence_role: str | None = None
    retrieval_tier: int | None = None
    landmark_acronym: str | None = None
    direction: ClaimDirection = "unclear"


@dataclass
class ClinicalClaim:
    """
    Afirmación clínica agregada (eje intervention × outcome × población), no un paper.

    ``contradicting`` lista PMIDs que matizan o contradicen el claim dominante.
    ``contradicts`` referencia otros ``claim_id`` en ejes relacionados (opcional).
    """

    claim_id: str
    axis_id: str
    axis_label: str = ""
    statement: str = ""
    landmark_support: list[str] = field(default_factory=list)
    intervention_concept_id: str | None = None
    comparator_concept_id: str | None = None
    outcome_concept_id: str | None = None
    population_concept_ids: list[str] = field(default_factory=list)
    direction: ClaimDirection = "unclear"
    evidence_strength: EvidenceStrength = "insufficient"
    consistency: ClaimConsistency = "insufficient"
    confidence: float = 0.0
    support: list[ClaimSupport] = field(default_factory=list)
    contradicting: list[ClaimSupport] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    applicability_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "axis_id": self.axis_id,
            "axis_label": self.axis_label or self.axis_id,
            "statement": self.statement,
            "landmark_support": list(self.landmark_support),
            "intervention_concept_id": self.intervention_concept_id,
            "comparator_concept_id": self.comparator_concept_id,
            "outcome_concept_id": self.outcome_concept_id,
            "population_concept_ids": list(self.population_concept_ids),
            "direction": self.direction,
            "evidence_strength": self.evidence_strength,
            "consistency": self.consistency,
            "confidence": round(self.confidence, 3),
            "support": [
                {
                    "pmid": s.pmid,
                    "study_type": s.study_type,
                    "evidence_role": s.evidence_role,
                    "retrieval_tier": s.retrieval_tier,
                    "landmark_acronym": s.landmark_acronym,
                    "direction": s.direction,
                }
                for s in self.support
            ],
            "contradicting": [
                {
                    "pmid": s.pmid,
                    "study_type": s.study_type,
                    "evidence_role": s.evidence_role,
                    "retrieval_tier": s.retrieval_tier,
                    "landmark_acronym": s.landmark_acronym,
                    "direction": s.direction,
                }
                for s in self.contradicting
            ],
            "contradicts": list(self.contradicts),
            "applicability_notes": list(self.applicability_notes),
        }


@dataclass
class ClaimBundle:
    """Conjunto de claims para un turno (entrada a síntesis claim-first)."""

    claims: list[ClinicalClaim] = field(default_factory=list)
    question_type: str = "general"
    unresolved_conflicts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "claims": [c.to_dict() for c in self.claims],
            "unresolved_conflicts": list(self.unresolved_conflicts),
        }
