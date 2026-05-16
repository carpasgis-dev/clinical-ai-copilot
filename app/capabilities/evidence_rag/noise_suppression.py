"""
Capa de supresión de ruido temático (soft penalty, sin exclusión dura).

Penaliza drift lateral solo cuando no forma parte del ``ClinicalIntent``.
Los temas estructurales se tratan en dos ejes independientes:

- **Translational** (murine, in vitro…): desajuste traslacional → penalización fuerte fija.
- **Evidence weakness** (commentary, editorial, case report…): baja jerarquía → penalización suave.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent

# (patrón, etiqueta) — orden: más específico primero.
_NEGATIVE_TOPIC_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\banimal\s+model\b", re.I), "animal model"),
    (re.compile(r"\bmurine\b|\bmice\b|\bmouse\b|\brats\b", re.I), "rodent model"),
    (re.compile(r"\bzebrafish\b|\bin\s+vitro\b|\bcell\s+line\b", re.I), "preclinical"),
    (re.compile(r"\bmolecular\s+mechanism|\bmechanistic\s+pathway|\bpathogenesis\b", re.I), "mechanistic"),
    (re.compile(r"\bnarrative\s+review\b", re.I), "narrative review"),
    (re.compile(r"\bcase\s+report\b", re.I), "case report"),
    (re.compile(r"\beditorial\b", re.I), "editorial"),
    (re.compile(r"\bletter\s+to\s+the\s+editor\b", re.I), "letter"),
    (re.compile(r"\bcommentary\b", re.I), "commentary"),
)

# Eje traslacional: un solo hit ya aplica el suelo (no acumular por len(topics) >= 2).
_TRANSLATIONAL_MULTIPLIERS: dict[str, float] = {
    "animal model": 0.25,
    "rodent model": 0.25,
    "preclinical": 0.25,
    "mechanistic": 0.40,
}

# Eje jerarquía de evidencia: penalización diferenciada (no toda la «mala» evidencia es igual).
_EVIDENCE_WEAK_MULTIPLIERS: dict[str, float] = {
    "narrative review": 0.85,
    "commentary": 0.82,
    "editorial": 0.75,
    "letter": 0.78,
    "case report": 0.45,
}

_INTENT_ALLOWED_TOPIC: dict[str, tuple[str, ...]] = {}


@dataclass(frozen=True, slots=True)
class StructuralNoiseResult:
    """Penalizaciones estructurales desacopladas (traslacional × evidencia débil)."""

    multiplier: float
    translational_penalty: float
    evidence_penalty: float
    matched_topics: tuple[str, ...]


# Alias retrocompatible con el nombre anterior en el pipeline.
NoiseSuppressionResult = StructuralNoiseResult


def _intent_blob(intent: ClinicalIntent | None) -> str:
    if intent is None:
        return ""
    parts: list[str] = []
    for xs in (
        intent.population,
        intent.interventions,
        intent.outcomes,
        intent.population_noise,
        intent.comparator,
    ):
        parts.extend(str(x) for x in xs)
    return fold_ascii(" ".join(parts))


def _topic_allowed_for_intent(topic: str, intent: ClinicalIntent | None) -> bool:
    if intent is None:
        return False
    blob = _intent_blob(intent)
    keys = _INTENT_ALLOWED_TOPIC.get(topic, ())
    if keys and any(k in blob for k in keys):
        return True
    if topic == "pregnancy" and "pregnancy" in [o.lower() for o in intent.population]:
        return True
    return False


def detect_negative_topics(blob: str) -> list[str]:
    """Temas estructurales detectados en título+abstract."""
    text = fold_ascii(blob or "")
    hits: list[str] = []
    for pat, label in _NEGATIVE_TOPIC_RULES:
        if pat.search(text):
            hits.append(label)
    return hits


def detect_negative_topics_for_intent(
    blob: str,
    *,
    clinical_intent: ClinicalIntent | None = None,
) -> list[str]:
    """Temas negativos tras filtrar los coherentes con el intent clínico."""
    raw = detect_negative_topics(blob)
    if not clinical_intent:
        return raw
    out: list[str] = []
    for label in raw:
        key = label.split()[0] if label else label
        if _topic_allowed_for_intent(key, clinical_intent):
            continue
        out.append(label)
    return out


def compute_structural_noise(
    topics: list[str],
) -> StructuralNoiseResult:
    """
    Combina ejes traslacional y de evidencia débil.

    Varios hits del mismo eje no empeoran más allá del mínimo (p. ej. in vitro + murine).
    """
    if not topics:
        return StructuralNoiseResult(
            multiplier=1.0,
            translational_penalty=1.0,
            evidence_penalty=1.0,
            matched_topics=(),
        )

    translational = 1.0
    evidence = 1.0
    for label in topics:
        if label in _TRANSLATIONAL_MULTIPLIERS:
            translational = min(translational, _TRANSLATIONAL_MULTIPLIERS[label])
        if label in _EVIDENCE_WEAK_MULTIPLIERS:
            evidence = min(evidence, _EVIDENCE_WEAK_MULTIPLIERS[label])

    combined = translational * evidence
    return StructuralNoiseResult(
        multiplier=combined,
        translational_penalty=translational,
        evidence_penalty=evidence,
        matched_topics=tuple(topics),
    )


def topic_drift_multiplier(
    title: str,
    abstract_snippet: str = "",
    *,
    clinical_intent: ClinicalIntent | None = None,
) -> float:
    """Multiplicador 0.25–1.0 sobre el score heurístico/semántico."""
    blob = f"{title} {abstract_snippet}"
    topics = detect_negative_topics_for_intent(blob, clinical_intent=clinical_intent)
    return compute_structural_noise(topics).multiplier


def apply_noise_suppression(
    score: float,
    title: str,
    abstract_snippet: str = "",
    *,
    clinical_intent: ClinicalIntent | None = None,
) -> tuple[float, StructuralNoiseResult]:
    """``score * multiplier`` con traza de temas y penalizaciones por eje."""
    blob = f"{title} {abstract_snippet}"
    topics = detect_negative_topics_for_intent(blob, clinical_intent=clinical_intent)
    ns = compute_structural_noise(topics)
    if not topics:
        return max(0.01, score), ns
    return max(0.01, score * ns.multiplier), ns
