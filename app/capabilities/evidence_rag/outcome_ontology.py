"""
Ontología de desenlaces clínicos: componentes compuestos (MACE) y expansión PubMed/semántica.

Reduce dependencia de keywords sueltos («cardiovascular», «MACE») ampliando endpoints
heterogéneos y paráfrasis frecuentes en trials y meta-análisis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent, OutcomeTheme, primary_outcome_theme

# ---------------------------------------------------------------------------
# Componentes atómicos y compuestos
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutcomeComponent:
    """Un desenlace o subcomponente de un endpoint compuesto."""

    key: str
    pubmed_tiab: tuple[str, ...]
    semantic_aliases: tuple[str, ...]


# Componentes típicos de MACE (3-point / 4-point)
_MACE_CV_DEATH = OutcomeComponent(
    key="cardiovascular_death",
    pubmed_tiab=(
        '"cardiovascular death"[tiab]',
        '"CV death"[tiab]',
        '"cardiac death"[tiab]',
        '"death from cardiovascular causes"[tiab]',
    ),
    semantic_aliases=("cardiovascular death", "cv death", "cardiac death"),
)
_MACE_NONFATAL_MI = OutcomeComponent(
    key="nonfatal_mi",
    pubmed_tiab=(
        '"nonfatal myocardial infarction"[tiab]',
        '"non-fatal myocardial infarction"[tiab]',
        '"nonfatal MI"[tiab]',
        '"myocardial infarction"[tiab]',
    ),
    semantic_aliases=("nonfatal myocardial infarction", "myocardial infarction", "mi"),
)
_MACE_NONFATAL_STROKE = OutcomeComponent(
    key="nonfatal_stroke",
    pubmed_tiab=(
        '"nonfatal stroke"[tiab]',
        '"non-fatal stroke"[tiab]',
        '"ischemic stroke"[tiab]',
        "stroke[tiab]",
    ),
    semantic_aliases=("nonfatal stroke", "ischemic stroke", "stroke"),
)
_MACE_COMPOSITE = OutcomeComponent(
    key="mace",
    pubmed_tiab=(
        '"major adverse cardiovascular events"[tiab]',
        "MACE[tiab]",
        '"3-point MACE"[tiab]',
        '"three-point MACE"[tiab]',
        '"4-point MACE"[tiab]',
        '"four-point MACE"[tiab]',
        '"composite cardiovascular outcome"[tiab]',
    ),
    semantic_aliases=("mace", "major adverse cardiovascular events"),
)

MACE_COMPONENTS: tuple[OutcomeComponent, ...] = (
    _MACE_COMPOSITE,
    _MACE_CV_DEATH,
    _MACE_NONFATAL_MI,
    _MACE_NONFATAL_STROKE,
)

_HF = OutcomeComponent(
    key="heart_failure",
    pubmed_tiab=(
        '"heart failure"[tiab]',
        '"heart failure hospitalization"[tiab]',
        '"hospitalization for heart failure"[tiab]',
        "HHF[tiab]",
    ),
    semantic_aliases=("heart failure", "hospitalization for heart failure", "hhf"),
)

_MORTALITY = OutcomeComponent(
    key="mortality",
    pubmed_tiab=(
        "mortality[tiab]",
        '"all-cause mortality"[tiab]',
        '"all cause mortality"[tiab]',
        '"cardiovascular mortality"[tiab]',
    ),
    semantic_aliases=("mortality", "all-cause mortality", "cardiovascular mortality"),
)

_CVOT = OutcomeComponent(
    key="cvot",
    pubmed_tiab=(
        "CVOT[tiab]",
        '"cardiovascular outcome trial"[tiab]',
        '"cardiovascular outcomes trial"[tiab]',
        '"outcomes trial"[tiab]',
    ),
    semantic_aliases=("cvot", "cardiovascular outcome trial"),
)

_CV_MODERATE = OutcomeComponent(
    key="cv_broad",
    pubmed_tiab=(
        '"cardiovascular benefit"[tiab]',
        '"cardiovascular risk"[tiab]',
        '"cardiovascular disease"[tiab]',
        '"cardiovascular events"[tiab]',
        '"heart disease"[tiab]',
    ),
    semantic_aliases=("cardiovascular benefit", "cardiovascular risk", "cardiovascular events"),
)

_RENAL = OutcomeComponent(
    key="renal",
    pubmed_tiab=(
        '"chronic kidney disease"[tiab]',
        "CKD[tiab]",
        '"renal outcomes"[tiab]',
        "eGFR[tiab]",
        "albuminuria[tiab]",
        "nephropathy[tiab]",
    ),
    semantic_aliases=("chronic kidney disease", "renal outcomes", "egfr", "albuminuria"),
)

_SAFETY = OutcomeComponent(
    key="safety",
    pubmed_tiab=(
        '"adverse events"[tiab]',
        "safety[tiab]",
        "hypoglycemia[tiab]",
        '"hypoglycaemia"[tiab]',
        "tolerability[tiab]",
    ),
    semantic_aliases=("adverse events", "safety", "hypoglycemia"),
)

_GLYCEMIC = OutcomeComponent(
    key="glycemic",
    pubmed_tiab=(
        "HbA1c[tiab]",
        '"glycemic control"[tiab]',
        "glycemia[tiab]",
    ),
    semantic_aliases=("hba1c", "glycemic control"),
)

_THEME_COMPONENTS: dict[OutcomeTheme, tuple[OutcomeComponent, ...]] = {
    "cv": (*MACE_COMPONENTS, _HF, _MORTALITY, _CVOT),
    "renal": (_RENAL,),
    "safety": (_SAFETY,),
    "glycemic": (_GLYCEMIC,),
    "general": (_MACE_COMPOSITE, _HF, _CV_MODERATE),
}


def _pubmed_or_from_components(components: tuple[OutcomeComponent, ...]) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for comp in components:
        for t in comp.pubmed_tiab:
            k = t.strip().lower()
            if k and k not in seen:
                seen.add(k)
                terms.append(t)
    if not terms:
        return ""
    return f"({' OR '.join(terms)})"


def pubmed_clause_cv_strict() -> str:
    """Endpoints CV duros + componentes MACE + CVOT."""
    return _pubmed_or_from_components(_THEME_COMPONENTS["cv"])


def pubmed_clause_cv_moderate() -> str:
    """Paráfrasis más amplias sin exigir MACE literal."""
    return _pubmed_or_from_components((*MACE_COMPONENTS[:1], _CV_MODERATE))


def pubmed_clause_cv_primary() -> str:
    """
    T1 (broad_primary): MACE + CVOT sin «heart disease» / «cardiovascular disease» genéricos.

    Reduce ``result_count`` en PubMed (objetivo ~5k–15k vs decenas de miles).
    """
    return _pubmed_or_from_components((*MACE_COMPONENTS, _CVOT))


def pubmed_clause_for_theme(theme: OutcomeTheme, *, tier: str) -> str | None:
    if tier == "broad":
        return None
    if theme == "cv":
        if tier == "strict":
            strict = pubmed_clause_cv_strict()
            return strict
        return pubmed_clause_cv_moderate()
    comps = _THEME_COMPONENTS.get(theme, _THEME_COMPONENTS["general"])
    if tier == "moderate" and theme == "general":
        return _pubmed_or_from_components((*comps, _CV_MODERATE))
    return _pubmed_or_from_components(comps)


def semantic_outcome_phrases_en(intent: ClinicalIntent | None) -> str:
    """Frase en inglés para bi-encoder / cross-encoder."""
    if intent is None:
        return _semantic_from_components(MACE_COMPONENTS + (_HF, _MORTALITY))
    theme = primary_outcome_theme(intent)
    comps = _THEME_COMPONENTS.get(theme, MACE_COMPONENTS)
    return _semantic_from_components(comps)


def _semantic_from_components(components: tuple[OutcomeComponent, ...]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for c in components:
        for a in c.semantic_aliases:
            if a not in seen:
                seen.add(a)
                parts.append(a)
    return ", ".join(parts[:24])


def outcome_component_hit(blob: str, component: OutcomeComponent) -> bool:
    text = (blob or "").lower()
    for alias in component.semantic_aliases:
        if alias.lower() in text:
            return True
    return False


def graded_cv_outcome_score(blob: str) -> float:
    """
    Score 0–1 por cobertura de componentes MACE/HF/mortalidad (alineación gradual).
    """
    if not blob.strip():
        return 0.0
    weights: list[tuple[OutcomeComponent, float]] = [
        (_MACE_COMPOSITE, 0.28),
        (_MACE_CV_DEATH, 0.22),
        (_MACE_NONFATAL_MI, 0.18),
        (_MACE_NONFATAL_STROKE, 0.18),
        (_HF, 0.14),
    ]
    score = 0.0
    for comp, w in weights:
        if outcome_component_hit(blob, comp):
            score += w
    return min(1.0, score)


# Patrones compilados para clinical_alignment (reexport)
CV_OUTCOME_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(re.escape(alias), re.I)
    for comp in MACE_COMPONENTS + (_HF, _MORTALITY)
    for alias in comp.semantic_aliases
)
