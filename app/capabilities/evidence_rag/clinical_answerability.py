"""
Utilidad clínica de un artículo para *soportar un claim* (no solo similitud semántica).

Scoring factorizado con ``AnswerabilityBreakdown`` para depuración.
Roles inferenciales alineados con ``evidence_role`` en rerank/UI.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from app.capabilities.clinical_sql.terminology import fold_ascii

if TYPE_CHECKING:
    from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent
    from app.capabilities.evidence_rag.clinical_semantics import ClinicalEvidenceFrame

# Jerarquía inferencial (peso para recomendación terapéutica)
EVIDENCE_ROLE_PRIMARY = "PRIMARY_THERAPEUTIC"
EVIDENCE_ROLE_SECONDARY = "SECONDARY_ANALYSIS"
EVIDENCE_ROLE_BACKGROUND = "BACKGROUND_CONTEXT"
EVIDENCE_ROLE_MECHANISTIC = "MECHANISTIC_SUPPORT"
EVIDENCE_ROLE_PRECLINICAL = "PRECLINICAL"

_ROLE_WEIGHT: dict[str, float] = {
    EVIDENCE_ROLE_PRIMARY: 1.0,
    EVIDENCE_ROLE_SECONDARY: 0.42,
    EVIDENCE_ROLE_BACKGROUND: 0.28,
    EVIDENCE_ROLE_MECHANISTIC: 0.18,
    EVIDENCE_ROLE_PRECLINICAL: 0.10,
}

_STRONG_PRIMARY_DESIGNS = frozenset(
    {
        "cvot-outcomes-trial",
        "rct",
        "meta-analysis",
        "network-meta-analysis",
        "systematic-review",
        "umbrella-review",
        "guideline",
    }
)

_NON_ANSWERABLE_DESIGNS = frozenset(
    {
        "basic-research",
        "preclinical",
        "editorial",
        "case-report",
        "case-series",
        "cross-sectional",
        "epidemiology",
    }
)

_WEAK_FOR_HEADLINES = frozenset(
    {
        "basic-research",
        "preclinical",
        "editorial",
        "case-report",
        "case-series",
        "cross-sectional",
        "epidemiology",
        "mechanistic-review",
        "narrative-review",
        "review",
    }
)

_SECONDARY_ANALYSIS = re.compile(
    r"\b("
    r"post[-\s]?hoc|subgroup analysis|sub-?group analysis|"
    r"secondary analysis|secondary endpoint|exploratory analysis|"
    r"prespecified secondary|sensitivity analysis|"
    r"ancillary study|substudy|sub-?study"
    r")\b",
    re.I,
)

_PRIMARY_TRIAL_SIGNAL = re.compile(
    r"\b("
    r"primary endpoint|primary outcome|randomi[sz]ed controlled trial|"
    r"double[-\s]?blind|placebo[-\s]?controlled|intention[-\s]?to[-\s]?treat|"
    r"cardiovascular outcomes trial|outcomes trial"
    r")\b",
    re.I,
)

_MECHANISM_ONLY = re.compile(
    r"\b(oxidative stress|dna damage|pathophysiology|mechanistic pathway|"
    r"in vitro|animal model|murine|fibrosis)\b",
    re.I,
)

_DESIGN_SIGNAL = re.compile(
    r"\b(randomi[sz]ed|controlled trial|rct|meta[-\s]?analysis|systematic review)\b",
    re.I,
)


@dataclass
class AnswerabilityBreakdown:
    """Componentes explícitos del score (trazabilidad / API debug)."""

    final_score: float = 0.0
    evidence_role: str = EVIDENCE_ROLE_BACKGROUND
    study_type: str | None = None
    design_base: float = 0.0
    evidence_role_weight: float = 0.0
    intervention_alignment: float = 0.0
    outcome_alignment: float = 0.0
    comparator_alignment: float = 0.0
    landmark_signal: float = 0.0
    topical_noise_penalty: float = 0.0
    secondary_analysis_penalty: float = 0.0
    mechanism_only_penalty: float = 0.0
    intervention_concept_ids: list[str] = field(default_factory=list)
    outcome_concept_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_question_type(query: str, intent: ClinicalIntent | None) -> str:
    from app.capabilities.evidence_rag.clinical_intent_graph import classify_question_type

    return classify_question_type(query, intent)


def infer_evidence_role(
    study_type: str | None,
    title: str,
    abstract: str,
) -> str:
    """
    Rol inferencial: primario vs secundario vs contexto.

    Distingue p. ej. «post hoc SUSTAIN-6» (SECONDARY) de CVOT primario (PRIMARY).
    """
    blob = f"{title} {abstract}"
    st = study_type or ""

    if st in ("basic-research", "preclinical"):
        return EVIDENCE_ROLE_PRECLINICAL
    if st in ("mechanistic-review",) or (
        _MECHANISM_ONLY.search(blob) and not _DESIGN_SIGNAL.search(blob)
    ):
        return EVIDENCE_ROLE_MECHANISTIC

    if _SECONDARY_ANALYSIS.search(blob):
        return EVIDENCE_ROLE_SECONDARY

    if st in ("narrative-review", "review", "editorial", "epidemiology", "cross-sectional"):
        return EVIDENCE_ROLE_BACKGROUND

    if st in _STRONG_PRIMARY_DESIGNS:
        if st in ("meta-analysis", "systematic-review", "network-meta-analysis", "umbrella-review"):
            if not _PRIMARY_TRIAL_SIGNAL.search(blob) and _SECONDARY_ANALYSIS.search(blob):
                return EVIDENCE_ROLE_SECONDARY
            return EVIDENCE_ROLE_PRIMARY
        if _PRIMARY_TRIAL_SIGNAL.search(blob) or st == "cvot-outcomes-trial":
            return EVIDENCE_ROLE_PRIMARY
        if st == "rct" and not _SECONDARY_ANALYSIS.search(blob):
            return EVIDENCE_ROLE_PRIMARY
        return EVIDENCE_ROLE_PRIMARY

    if st in ("clinical-trial", "pilot-trial", "cohort"):
        return EVIDENCE_ROLE_BACKGROUND

    return EVIDENCE_ROLE_BACKGROUND


def evidence_role_label(
    study_type: str | None,
    title: str = "",
    abstract: str = "",
) -> str:
    """Etiqueta corta para rerank/UI (derivada del rol inferencial)."""
    role = infer_evidence_role(study_type, title, abstract)
    mapping = {
        EVIDENCE_ROLE_PRIMARY: "PRIMARY_THERAPEUTIC",
        EVIDENCE_ROLE_SECONDARY: "SECONDARY_ANALYSIS",
        EVIDENCE_ROLE_BACKGROUND: "BACKGROUND",
        EVIDENCE_ROLE_MECHANISTIC: "MECHANISTIC",
        EVIDENCE_ROLE_PRECLINICAL: "PRECLINICAL",
    }
    return mapping.get(role, "UNKNOWN")


def _design_base_score(study_type: str | None, evidence_role: str) -> float:
    st = study_type or ""
    if evidence_role == EVIDENCE_ROLE_PRECLINICAL:
        return 0.12
    if evidence_role == EVIDENCE_ROLE_MECHANISTIC:
        return 0.20
    if evidence_role == EVIDENCE_ROLE_SECONDARY:
        return 0.52
    if evidence_role == EVIDENCE_ROLE_BACKGROUND:
        return 0.32
    if st == "cvot-outcomes-trial":
        return 0.95
    if st == "rct":
        return 0.88
    if st in ("meta-analysis", "network-meta-analysis", "umbrella-review"):
        return 0.80
    if st == "systematic-review":
        return 0.78
    if st == "guideline":
        return 0.82
    if st in ("clinical-trial", "pilot-trial"):
        return 0.55
    if st == "cohort":
        return 0.48
    return 0.40


def _topical_noise_hit(blob: str) -> bool:
    from app.capabilities.evidence_rag.clinical_semantics import CLINICAL_CONCEPTS

    for c in CLINICAL_CONCEPTS.values():
        if c.kind != "suppress_category":
            continue
        if c.matches_text(blob):
            if c.id == "suppress_topical_report" and _PRIMARY_TRIAL_SIGNAL.search(blob):
                continue
            return True
    return False


def _alignment_from_intent(
    blob: str,
    intent: ClinicalIntent | None,
    *,
    frame: ClinicalEvidenceFrame | None = None,
) -> tuple[float, float, float, list[str], list[str]]:
    """intervention, outcome, comparator alignment 0–1."""
    from app.capabilities.evidence_rag.clinical_semantics import (
        classify_intervention_in_text,
        get_concept,
    )

    iv_ids: list[str] = []
    if frame and frame.intervention_concept_ids:
        iv_ids = list(frame.intervention_concept_ids)
    else:
        iv_ids = classify_intervention_in_text(blob)

    out_ids: list[str] = []
    if frame and frame.outcome_concept_ids:
        for oid in frame.outcome_concept_ids:
            c = get_concept(oid)
            if c and c.matches_text(blob):
                out_ids.append(oid)
    elif intent:
        for label in intent.outcomes:
            cid = None
            from app.capabilities.evidence_rag.clinical_semantics import resolve_concept_id

            cid = resolve_concept_id(label)
            if cid:
                c = get_concept(cid)
                if c and c.matches_text(blob):
                    out_ids.append(cid)

    iv_align = 0.0
    if not intent or not intent.interventions:
        iv_align = 0.55 if iv_ids else 0.45
    elif iv_ids:
        iv_align = min(1.0, 0.65 + 0.12 * len(iv_ids))
    else:
        iv_align = 0.25

    out_align = 0.0
    if not intent or not intent.outcomes:
        out_align = 0.50 if out_ids else 0.45
    elif out_ids:
        out_align = min(1.0, 0.60 + 0.15 * len(out_ids))
    else:
        out_align = 0.30

    comp_align = 0.55
    if intent and intent.comparator:
        comp_hit = any(
            fold_ascii(x)[:6] in blob for x in intent.comparator if len(x) > 2
        )
        iv_hit = iv_align >= 0.5
        if comp_hit and iv_hit:
            comp_align = 0.92
        elif comp_hit or iv_hit:
            comp_align = 0.62
        else:
            comp_align = 0.28
    return iv_align, out_align, comp_align, iv_ids, out_ids


def _landmark_signal(title: str, abstract: str, evidence_role: str) -> float:
    from app.capabilities.evidence_rag.landmark_registry import match_landmark_trial

    trial = match_landmark_trial(title, abstract)
    if not trial:
        return 0.0
    if evidence_role == EVIDENCE_ROLE_PRIMARY:
        return 0.95
    if evidence_role == EVIDENCE_ROLE_SECONDARY:
        return 0.35
    return 0.20


def compute_answerability_breakdown(
    study_type: str | None,
    title: str,
    abstract: str,
    *,
    intent: ClinicalIntent | None = None,
    question_type: str | None = None,
    frame: ClinicalEvidenceFrame | None = None,
) -> AnswerabilityBreakdown:
    """
    Score factorizado + rol inferencial.

    ``final_score`` ≈ diseño × rol × alineaciones × penalizaciones.
    """
    blob = fold_ascii(f"{title} {abstract}")
    st = study_type or ""
    qtype = question_type or (infer_question_type("", intent) if intent else "general")
    therapeutic_q = qtype in (
        "comparative_effectiveness",
        "treatment_efficacy",
        "treatment_selection",
    )

    role = infer_evidence_role(st, title, abstract)
    role_w = _ROLE_WEIGHT.get(role, 0.28)
    design = _design_base_score(st, role)

    iv_a, out_a, comp_a, iv_ids, out_ids = _alignment_from_intent(
        blob, intent, frame=frame
    )
    lm = _landmark_signal(title, abstract, role)

    topical_pen = 1.0
    if therapeutic_q and _topical_noise_hit(blob):
        topical_pen = 0.22

    secondary_pen = 1.0
    if role == EVIDENCE_ROLE_SECONDARY:
        secondary_pen = 0.55

    mechanism_pen = 1.0
    if _MECHANISM_ONLY.search(blob) and not _DESIGN_SIGNAL.search(blob):
        mechanism_pen = 0.18

    align_mix = (
        0.30 * iv_a
        + 0.28 * out_a
        + 0.12 * comp_a
        + 0.30 * max(iv_a, out_a, lm)
    )

    raw = design * role_w * (0.55 + 0.45 * align_mix)
    raw *= topical_pen * secondary_pen * mechanism_pen

    if therapeutic_q and role == EVIDENCE_ROLE_PRIMARY and lm >= 0.9:
        raw = min(1.0, raw + 0.08)

    if therapeutic_q and role == EVIDENCE_ROLE_SECONDARY:
        raw = min(raw, 0.48)

    if (
        therapeutic_q
        and intent
        and intent.comparator
        and intent.interventions
        and comp_a < 0.35
        and iv_a < 0.35
        and role not in (EVIDENCE_ROLE_PRIMARY,)
    ):
        raw = min(raw, 0.32)

    final = max(0.05, min(1.0, raw))

    return AnswerabilityBreakdown(
        final_score=round(final, 4),
        evidence_role=role,
        study_type=st or None,
        design_base=round(design, 4),
        evidence_role_weight=round(role_w, 4),
        intervention_alignment=round(iv_a, 4),
        outcome_alignment=round(out_a, 4),
        comparator_alignment=round(comp_a, 4),
        landmark_signal=round(lm, 4),
        topical_noise_penalty=round(topical_pen, 4),
        secondary_analysis_penalty=round(secondary_pen, 4),
        mechanism_only_penalty=round(mechanism_pen, 4),
        intervention_concept_ids=iv_ids,
        outcome_concept_ids=out_ids,
    )


def clinical_answerability_score(
    study_type: str | None,
    title: str,
    abstract: str,
    *,
    intent: ClinicalIntent | None = None,
    question_type: str | None = None,
    frame: ClinicalEvidenceFrame | None = None,
) -> float:
    """0–1: utilidad para soportar un claim clínico en la pregunta."""
    return compute_answerability_breakdown(
        study_type,
        title,
        abstract,
        intent=intent,
        question_type=question_type,
        frame=frame,
    ).final_score


def passes_answerability_gate(
    title: str,
    abstract: str,
    *,
    study_type: str | None = None,
    intent: ClinicalIntent | None = None,
    question_type: str | None = None,
    frame: ClinicalEvidenceFrame | None = None,
) -> bool:
    """Filtro duro pre-rerank."""
    bd = compute_answerability_breakdown(
        study_type,
        title,
        abstract,
        intent=intent,
        question_type=question_type,
        frame=frame,
    )
    if bd.final_score < 0.22:
        return False
    if bd.topical_noise_penalty < 0.5 and bd.landmark_signal < 0.3:
        return False
    if bd.mechanism_only_penalty < 0.5:
        return False
    return True


def eligible_for_featured_headline(
    study_type: str | None,
    title: str,
    abstract: str,
    *,
    pool_has_strong_answerable: bool,
    intent: ClinicalIntent | None = None,
    frame: ClinicalEvidenceFrame | None = None,
) -> bool:
    """Referencias destacadas: solo evidencia primaria o secundaria con señal fuerte."""
    bd = compute_answerability_breakdown(
        study_type, title, abstract, intent=intent, frame=frame
    )
    if bd.final_score < 0.42:
        return False
    if bd.evidence_role in (EVIDENCE_ROLE_PRECLINICAL, EVIDENCE_ROLE_MECHANISTIC):
        return False
    if pool_has_strong_answerable:
        if bd.evidence_role in (EVIDENCE_ROLE_BACKGROUND, EVIDENCE_ROLE_SECONDARY):
            if bd.landmark_signal < 0.5 and bd.final_score < 0.55:
                return False
        if study_type in _WEAK_FOR_HEADLINES and bd.evidence_role != EVIDENCE_ROLE_PRIMARY:
            return False
    if study_type in _NON_ANSWERABLE_DESIGNS:
        return False
    return True
