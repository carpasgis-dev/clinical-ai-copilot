"""Tiers epistémicos de recuperación (1 = PICO exacto … 5 = mecanístico/contextual)."""
from __future__ import annotations

from enum import IntEnum


class RetrievalTier(IntEnum):
    T1_EXACT_PICO = 1
    T2_LANDMARK_CVOT = 2
    T3_SYSTEMATIC = 3
    T4_SEMANTIC_BROAD = 4
    T5_MECHANISTIC = 5


# Multiplicador de score post-heurística / semántica (menor tier → mayor peso).
_TIER_WEIGHT: dict[int, float] = {
    int(RetrievalTier.T1_EXACT_PICO): 1.0,
    int(RetrievalTier.T2_LANDMARK_CVOT): 0.95,
    int(RetrievalTier.T3_SYSTEMATIC): 0.88,
    int(RetrievalTier.T4_SEMANTIC_BROAD): 0.72,
    int(RetrievalTier.T5_MECHANISTIC): 0.55,
}


def tier_weight_multiplier(tier: int | None) -> float:
    """Peso 0.55–1.0 según tier de recuperación; desconocido → 0.65."""
    if tier is None:
        return 0.65
    try:
        t = int(tier)
    except (TypeError, ValueError):
        return 0.65
    if t >= 99:
        return 0.65
    return _TIER_WEIGHT.get(t, 0.65)


def tier_short_label(tier: int) -> str:
    labels = {
        1: "PICO exacto",
        2: "landmark/CVOT",
        3: "revisión sistemática",
        4: "expansión semántica",
        5: "mecanístico/contextual",
    }
    return labels.get(int(tier), f"tier {tier}")


def tier_retrieval_provenance_line(tier: int, stage: str = "") -> str | None:
    """
    Frase obligatoria bajo ### PMID cuando tier > 1 (síntesis tier-aware).
    """
    try:
        t = int(tier)
    except (TypeError, ValueError):
        return None
    if t <= 1:
        return None
    stage_bit = f" (etapa: {stage})" if stage.strip() else ""
    if t == 2:
        return (
            f"*Recuperado vía ensayo landmark o CVOT (tier 2{stage_bit}); "
            "puede extrapolarse a la población preguntada con cautela.*"
        )
    if t == 3:
        return (
            f"*Recuperado vía revisión/meta-análisis (tier 3{stage_bit}); "
            "síntesis indirecta, no necesariamente RCT en la cohorte exacta.*"
        )
    if t >= 4:
        return (
            f"*Recuperado vía expansión de búsqueda (tier {t}{stage_bit}); "
            "evidencia contextual — no afirmar eficacia clínica directa sin leer el abstract.*"
        )
    return None
