"""
Extracción determinista de claims clínicos desde artículos rankeados.

Slice inicial: SGLT2, GLP-1 RA, DOAC vs warfarina (CV / anticoagulación).
Sin LLM: concept matching + ejes de desenlace + direccionalidad léxica.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.capabilities.evidence_rag.clinical_answerability import (
    EVIDENCE_ROLE_BACKGROUND,
    EVIDENCE_ROLE_MECHANISTIC,
    EVIDENCE_ROLE_PRECLINICAL,
    EVIDENCE_ROLE_PRIMARY,
    EVIDENCE_ROLE_SECONDARY,
    compute_answerability_breakdown,
    infer_evidence_role,
)
from app.capabilities.evidence_rag.clinical_claims import (
    ClaimBundle,
    ClaimConsistency,
    ClaimDirection,
    ClaimSupport,
    ClinicalClaim,
    EvidenceStrength,
)
from app.capabilities.evidence_rag.clinical_semantics import (
    ClinicalEvidenceFrame,
    frame_from_state,
    get_concept,
    policy_for_question_type,
)
from app.capabilities.evidence_rag.evidence_rerank import infer_study_type_from_title
from app.capabilities.evidence_rag.landmark_registry import match_landmark_trial

_BENEFIT = re.compile(
    r"\b("
    r"reduced risk|reduction in risk|lower risk|decreased risk|"
    r"reduced incidence|lower incidence|reduced rate|"
    r"benefit|favorable|superior|noninferior|non-inferior|"
    r"improved outcomes|fewer (events|hospitalizations)"
    r")\b",
    re.I,
)
_HARM = re.compile(
    r"\b(increased risk|higher risk|worse outcomes|inferior to|caused harm)\b",
    re.I,
)
_NEUTRAL = re.compile(
    r"\b("
    r"no significant difference|not significant|similar risk|comparable|"
    r"no difference in|did not reduce|failed to show"
    r")\b",
    re.I,
)

_STRONG_DESIGNS = frozenset(
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

_WEAK_DESIGNS = frozenset(
    {
        "narrative-review",
        "review",
        "editorial",
        "epidemiology",
        "cross-sectional",
        "cohort",
    }
)

_TOPICAL_ONLY = re.compile(
    r"\b(summit report|conference report|year in review|position paper)\b",
    re.I,
)

_LANDMARK_CLASS_TO_CONCEPT: dict[str, str] = {
    "sglt2": "sglt2_class",
    "glp1": "glp1_ra",
    "doac": "doac_class",
}

# Slice v1: solo tres ejes terapéuticos (no generalizar aún).
PRIMARY_SLICE_AXIS_IDS: frozenset[str] = frozenset(
    {
        "sglt2_hf_hospitalization",
        "glp1_ra_mace",
        "doac_vs_warfarin_ic_bleeding",
    }
)

AXIS_DISPLAY_LABELS: dict[str, str] = {
    "sglt2_hf_hospitalization": "SGLT2 → hospitalización por insuficiencia cardíaca",
    "glp1_ra_mace": "GLP-1 RA → MACE (eventos cardiovasculares mayores)",
    "doac_vs_warfarin_ic_bleeding": "DOAC vs warfarina → hemorragia intracraneal",
}

_DIRECTION_GLYPH: dict[str, str] = {
    "benefit": "↓",
    "harm": "↑",
    "neutral": "↔",
    "unclear": "?",
}

_POP_LABELS: dict[str, str] = {
    "pop_t2dm": "DM2",
    "pop_af": "fibrilación auricular",
    "pop_older_adults": "adultos mayores (≥65)",
}


def primary_slice_enabled() -> bool:
    """
    ``COPILOT_CLAIM_SLICE=1`` (default): solo 3 ejes (SGLT2/HF, GLP-1/MACE, DOAC/IC bleed).
    ``0`` / ``all`` / ``full``: todos los ejes definidos en ``_CLAIM_AXES``.
    """
    raw = (os.getenv("COPILOT_CLAIM_SLICE") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "all", "full")


def _axis_label_for(axis_id: str, direction: ClaimDirection) -> str:
    base = AXIS_DISPLAY_LABELS.get(axis_id, axis_id)
    glyph = _DIRECTION_GLYPH.get(direction, "?")
    if "→" in base:
        left, right = base.split("→", 1)
        return f"{left.strip()} {glyph} {right.strip()}"
    return f"{base} {glyph}"


@dataclass(frozen=True, slots=True)
class _ClaimAxis:
    axis_id: str
    intervention_id: str
    outcome_id: str | None
    comparator_id: str | None = None
    population_ids: tuple[str, ...] = ()
    outcome_extra_pattern: str | None = None
    requires_comparator: bool = False
    statement_benefit: str = ""
    statement_neutral: str = ""
    statement_harm: str = ""


_CLAIM_AXES: tuple[_ClaimAxis, ...] = (
    _ClaimAxis(
        axis_id="sglt2_hf_hospitalization",
        intervention_id="sglt2_class",
        outcome_id="outcome_hf_hosp",
        population_ids=("pop_t2dm",),
        statement_benefit=(
            "Los inhibidores SGLT2 reducen hospitalización por insuficiencia cardíaca "
            "en población con diabetes tipo 2 (evidencia de ensayos de desenlace/cardiorenal)."
        ),
        statement_neutral=(
            "Evidencia mixta o no concluyente sobre hospitalización por IC con SGLT2 en el pool."
        ),
    ),
    _ClaimAxis(
        axis_id="sglt2_mace",
        intervention_id="sglt2_class",
        outcome_id="outcome_mace",
        population_ids=("pop_t2dm",),
        statement_benefit=(
            "Los SGLT2 muestran beneficio modesto o neutralidad favorable en MACE/eventos CV mayores "
            "según CVOTs (p. ej. EMPA-REG, DECLARE, CANVAS), con énfasis cardiorenal."
        ),
        statement_neutral="Efecto en MACE con SGLT2 no uniforme o no estadísticamente claro en el pool.",
    ),
    _ClaimAxis(
        axis_id="sglt2_cardiorenal",
        intervention_id="sglt2_class",
        outcome_id=None,
        outcome_extra_pattern=r"\b(cardiorenal|kidney|renal|nephropathy|albuminuria)\b",
        population_ids=("pop_t2dm",),
        statement_benefit=(
            "Los SGLT2 aportan beneficio cardiorenal (progresión renal / desenlaces renales compuestos)."
        ),
    ),
    _ClaimAxis(
        axis_id="glp1_ra_mace",
        intervention_id="glp1_ra",
        outcome_id="outcome_mace",
        population_ids=("pop_t2dm",),
        statement_benefit=(
            "Los agonistas GLP-1 RA reducen MACE/eventos cardiovasculares mayores en ensayos "
            "como LEADER, SUSTAIN-6 o REWIND (según extractos indexados)."
        ),
        statement_neutral="Beneficio en MACE con GLP-1 RA no claro en todas las referencias del pool.",
    ),
    _ClaimAxis(
        axis_id="glp1_ra_weight",
        intervention_id="glp1_ra",
        outcome_id=None,
        outcome_extra_pattern=r"\b(weight loss|weight reduction|body weight|reduction in weight)\b",
        population_ids=("pop_t2dm",),
        statement_benefit="Los GLP-1 RA se asocian a reducción de peso corporal en los estudios indexados.",
    ),
    _ClaimAxis(
        axis_id="glp1_ra_ascvd",
        intervention_id="glp1_ra",
        outcome_id="outcome_mace",
        outcome_extra_pattern=r"\b(ascvd|atherosclerotic cardiovascular)\b",
        population_ids=("pop_t2dm",),
        statement_benefit=(
            "Los GLP-1 RA muestran beneficio en prevención de enfermedad cardiovascular aterosclerótica (ASCVD)."
        ),
    ),
    _ClaimAxis(
        axis_id="doac_vs_warfarin_ic_bleeding",
        intervention_id="doac_class",
        comparator_id="warfarin",
        outcome_id="outcome_bleeding",
        population_ids=("pop_af",),
        requires_comparator=True,
        statement_benefit=(
            "Los DOAC reducen hemorragia intracraneal / hemorragia mayor frente a warfarina "
            "en fibrilación auricular (ARISTOTLE, RE-LY, ROCKET AF, ENGAGE AF según pool)."
        ),
        statement_neutral=(
            "Comparación DOAC vs warfarina en sangrado: sin diferencia clara en parte del pool."
        ),
    ),
    _ClaimAxis(
        axis_id="doac_vs_warfarin_stroke",
        intervention_id="doac_class",
        comparator_id="warfarin",
        outcome_id="outcome_stroke",
        population_ids=("pop_af",),
        requires_comparator=True,
        statement_benefit=(
            "Los DOAC son no inferiores o superiores a warfarina en prevención de ictus/embolia "
            "en FA no valvular según ensayos landmark del pool."
        ),
        statement_neutral="Eficacia anticoagulante DOAC vs warfarina en ictus: resultados heterogéneos en el pool.",
    ),
    _ClaimAxis(
        axis_id="doac_guideline_preference_nvaf",
        intervention_id="doac_class",
        comparator_id="warfarin",
        outcome_id=None,
        outcome_extra_pattern=r"\b(guideline|recommended|preferred|class i)\b",
        population_ids=("pop_af",),
        requires_comparator=True,
        statement_benefit=(
            "Guías y revisiones del pool favorecen DOAC frente a warfarina en FA no valvular "
            "cuando no hay contraindicación."
        ),
    ),
)


@dataclass
class _ArticleHit:
    pmid: str
    title: str
    study_type: str | None
    evidence_role: str
    direction: ClaimDirection
    landmark: str | None
    answerability: float
    is_weak: bool


def _intervention_matches(intervention_id: str, title: str, snip: str) -> bool:
    blob = fold_ascii(f"{title} {snip}")
    iv = get_concept(intervention_id)
    if iv and iv.matches_text(blob):
        return True
    trial = match_landmark_trial(title, snip)
    if trial:
        for dc in trial.drug_classes:
            if _LANDMARK_CLASS_TO_CONCEPT.get(dc) == intervention_id:
                return True
    return False


def _eligible_for_claim(
    title: str,
    abstract: str,
    *,
    study_type: str | None,
    question_type: str | None,
    frame: ClinicalEvidenceFrame | None,
) -> bool:
    """Filtro ligero: ruido tópico/mecanismo; no exige score alto de answerability."""
    blob = fold_ascii(f"{title} {abstract}")
    if _TOPICAL_ONLY.search(blob) and not re.search(
        r"\b(randomi[sz]ed|meta[-\s]?analysis|hazard ratio)\b", blob, re.I
    ):
        return False
    bd = compute_answerability_breakdown(
        study_type,
        title,
        abstract,
        question_type=question_type,
        frame=frame,
    )
    if bd.mechanism_only_penalty < 0.5:
        return False
    if bd.topical_noise_penalty < 0.5:
        return False
    if bd.evidence_role == EVIDENCE_ROLE_PRECLINICAL:
        return False
    return True


def _detect_direction(blob: str) -> ClaimDirection:
    if _NEUTRAL.search(blob):
        return "neutral"
    if _HARM.search(blob):
        return "harm"
    if _BENEFIT.search(blob):
        return "benefit"
    return "unclear"


def _outcome_matches(axis: _ClaimAxis, blob: str) -> bool:
    if axis.outcome_id:
        c = get_concept(axis.outcome_id)
        if c and c.matches_text(blob):
            return True
    if axis.outcome_extra_pattern and re.search(axis.outcome_extra_pattern, blob, re.I):
        return True
    return axis.outcome_id is None and axis.outcome_extra_pattern is None


def _population_ids(blob: str, axis: _ClaimAxis) -> list[str]:
    found: list[str] = []
    for pid in axis.population_ids:
        c = get_concept(pid)
        if c and c.matches_text(blob):
            found.append(pid)
    return found


def _article_hits_axis(
    axis: _ClaimAxis,
    *,
    title: str,
    snip: str,
    pmid: str,
    question_type: str | None,
    frame: ClinicalEvidenceFrame | None = None,
) -> _ArticleHit | None:
    blob = fold_ascii(f"{title} {snip}")
    if _TOPICAL_ONLY.search(blob) and not re.search(
        r"\b(randomi[sz]ed|meta[-\s]?analysis|hazard ratio)\b", blob, re.I
    ):
        return None

    if not _intervention_matches(axis.intervention_id, title, snip):
        return None

    if axis.requires_comparator:
        comp = get_concept(axis.comparator_id or "")
        if not comp or not comp.matches_text(blob):
            return None

    if not _outcome_matches(axis, blob):
        return None

    st = infer_study_type_from_title(title)
    if not _eligible_for_claim(
        title, snip, study_type=st, question_type=question_type, frame=frame
    ):
        return None

    role = infer_evidence_role(st, title, snip)
    bd = compute_answerability_breakdown(
        st, title, snip, question_type=question_type, frame=frame
    )
    trial = match_landmark_trial(title, snip)
    direction = _detect_direction(blob)
    if direction == "unclear" and role == EVIDENCE_ROLE_PRIMARY:
        direction = "benefit"

    weak = (
        role in (EVIDENCE_ROLE_BACKGROUND, EVIDENCE_ROLE_SECONDARY, EVIDENCE_ROLE_MECHANISTIC)
        or st in _WEAK_DESIGNS
        or role == EVIDENCE_ROLE_PRECLINICAL
    )

    return _ArticleHit(
        pmid=pmid,
        title=title[:120],
        study_type=st,
        evidence_role=role,
        direction=direction,
        landmark=trial.acronym if trial else None,
        answerability=bd.final_score,
        is_weak=weak,
    )


def _hit_to_support(hit: _ArticleHit, art: dict[str, Any]) -> ClaimSupport:
    tier = art.get("retrieval_tier")
    return ClaimSupport(
        pmid=hit.pmid,
        study_type=hit.study_type,
        evidence_role=hit.evidence_role,
        retrieval_tier=int(tier) if tier is not None else None,
        landmark_acronym=hit.landmark,
        direction=hit.direction,
    )


def _strength_and_consistency(
    supporting: list[_ArticleHit],
    contradicting: list[_ArticleHit],
) -> tuple[EvidenceStrength, ClaimConsistency, ClaimDirection, float]:
    if not supporting:
        return "insufficient", "insufficient", "unclear", 0.0

    primary = [h for h in supporting if h.evidence_role == EVIDENCE_ROLE_PRIMARY]
    landmarks = [h for h in supporting if h.landmark]
    strong = [h for h in supporting if h.study_type in _STRONG_DESIGNS]

    dirs = [h.direction for h in supporting if h.direction != "unclear"]
    dom: ClaimDirection = "benefit"
    if dirs:
        benefit_n = sum(1 for d in dirs if d == "benefit")
        neutral_n = sum(1 for d in dirs if d == "neutral")
        harm_n = sum(1 for d in dirs if d == "harm")
        if harm_n > benefit_n and harm_n >= neutral_n:
            dom = "harm"
        elif neutral_n > benefit_n:
            dom = "neutral"
        elif benefit_n >= neutral_n:
            dom = "benefit"
        else:
            dom = "unclear"

    consistency: ClaimConsistency = "consistent"
    if contradicting:
        opp = [c for c in contradicting if c.direction != "unclear" and c.direction != dom]
        if opp:
            consistency = "conflicting" if any(not c.is_weak for c in contradicting) else "mixed"
        else:
            consistency = "mixed"
    elif len(set(dirs)) > 1:
        consistency = "mixed"

    if len(primary) >= 2 or (len(primary) >= 1 and len(landmarks) >= 2):
        strength: EvidenceStrength = "high"
    elif len(primary) >= 1 or len(landmarks) >= 1 or len(strong) >= 2:
        strength = "moderate"
    elif len(strong) >= 1 or len(supporting) >= 2:
        strength = "low"
    else:
        strength = "insufficient"

    conf = 0.35
    if strength == "high":
        conf = 0.88 if consistency == "consistent" else 0.72
    elif strength == "moderate":
        conf = 0.72 if consistency == "consistent" else 0.58
    elif strength == "low":
        conf = 0.48
    else:
        conf = 0.28

    return strength, consistency, dom, conf


def _applicability_from_frame(
    frame: ClinicalEvidenceFrame | None,
    pop_ids: list[str],
) -> list[str]:
    notes: list[str] = []
    for pid in pop_ids:
        if pid in _POP_LABELS:
            notes.append(_POP_LABELS[pid])
    if frame:
        for pid in frame.population_concept_ids:
            if pid in _POP_LABELS and _POP_LABELS[pid] not in notes:
                notes.append(_POP_LABELS[pid])
        if frame.age_min is not None and frame.age_min >= 65:
            if "adultos mayores (≥65)" not in notes:
                notes.append("adultos mayores (≥65)")
    return notes


def _axes_for_frame(
    frame: ClinicalEvidenceFrame | None,
    *,
    primary_slice: bool,
) -> tuple[_ClaimAxis, ...]:
    pool = _primary_slice_axes() if primary_slice else _CLAIM_AXES
    if frame is None:
        return pool

    iv_set = set(frame.intervention_concept_ids)
    comp_set = set(frame.comparator_concept_ids)
    out_set = set(frame.outcome_concept_ids)

    if not iv_set and not comp_set:
        return pool

    selected: list[_ClaimAxis] = []
    for axis in pool:
        if iv_set and axis.intervention_id not in iv_set:
            if not (axis.requires_comparator and "doac_class" in iv_set):
                continue
        if axis.requires_comparator and comp_set and "warfarin" not in comp_set:
            if "doac_class" not in iv_set:
                continue
        if out_set and axis.outcome_id and axis.outcome_id not in out_set:
            if axis.axis_id not in (
                "sglt2_hf_hospitalization",
                "sglt2_cardiorenal",
                "glp1_ra_weight",
            ):
                continue
        selected.append(axis)

    return tuple(selected) if selected else pool


def _primary_slice_axes() -> tuple[_ClaimAxis, ...]:
    return tuple(a for a in _CLAIM_AXES if a.axis_id in PRIMARY_SLICE_AXIS_IDS)


def extract_claims_deterministic(
    frame: ClinicalEvidenceFrame | None,
    articles: list[dict[str, Any]],
    *,
    question_type: str | None = None,
    primary_slice: bool = True,
) -> ClaimBundle:
    """
    Agrupa artículos rankeados en claims por ``axis_id`` (intervención × desenlace × población).
    """
    qtype = question_type or (frame.question_type if frame else "general")
    policy = policy_for_question_type(qtype)
    if not policy.therapeutic_objective and not articles:
        return ClaimBundle(question_type=qtype)

    sorted_arts = sorted(
        [a for a in articles if isinstance(a, dict) and str(a.get("pmid") or "").strip()],
        key=lambda a: float(a.get("final_rank_score") or 0.0),
        reverse=True,
    )

    axes = _axes_for_frame(frame, primary_slice=primary_slice)
    claims: list[ClinicalClaim] = []
    unresolved: list[str] = []

    art_by_pmid = {str(a.get("pmid")): a for a in sorted_arts}

    for axis in axes:
        supporting_hits: list[_ArticleHit] = []
        contradict_hits: list[_ArticleHit] = []

        for art in sorted_arts:
            pmid = str(art.get("pmid") or "").strip()
            title = str(art.get("title") or "")
            snip = str(art.get("abstract_snippet") or "")
            hit = _article_hits_axis(
                axis,
                title=title,
                snip=snip,
                pmid=pmid,
                question_type=qtype,
                frame=frame,
            )
            if not hit:
                continue
            if hit.is_weak and hit.direction in ("neutral", "harm"):
                contradict_hits.append(hit)
            elif (
                not hit.is_weak
                and hit.direction in ("neutral", "harm")
                and supporting_hits
                and any(h.direction == "benefit" for h in supporting_hits)
            ):
                contradict_hits.append(hit)
            else:
                supporting_hits.append(hit)

        if not supporting_hits:
            continue

        strength, consistency, dom, conf = _strength_and_consistency(
            supporting_hits, contradict_hits
        )
        if dom == "benefit":
            statement = axis.statement_benefit
        elif dom == "neutral" and axis.statement_neutral:
            statement = axis.statement_neutral
        elif dom == "harm":
            statement = axis.statement_benefit.replace("reducen", "podrían aumentar").replace(
                "beneficio", "riesgo"
            )
        else:
            statement = axis.statement_benefit

        pop_ids = []
        for h in supporting_hits[:3]:
            art = art_by_pmid.get(h.pmid) or {}
            pop_ids.extend(
                _population_ids(
                    fold_ascii(
                        f"{art.get('title') or ''} {art.get('abstract_snippet') or ''}"
                    ),
                    axis,
                )
            )
        pop_ids = list(dict.fromkeys(pop_ids))
        applicability = _applicability_from_frame(frame, pop_ids)
        landmarks = sorted({h.landmark for h in supporting_hits if h.landmark})

        claim = ClinicalClaim(
            claim_id=axis.axis_id,
            axis_id=axis.axis_id,
            axis_label=_axis_label_for(axis.axis_id, dom),
            statement=statement,
            landmark_support=landmarks,
            intervention_concept_id=axis.intervention_id,
            comparator_concept_id=axis.comparator_id,
            outcome_concept_id=axis.outcome_id,
            population_concept_ids=pop_ids,
            direction=dom,
            evidence_strength=strength,
            consistency=consistency,
            confidence=conf,
            support=[_hit_to_support(h, art_by_pmid.get(h.pmid) or {}) for h in supporting_hits[:8]],
            contradicting=[
                _hit_to_support(h, art_by_pmid.get(h.pmid) or {}) for h in contradict_hits[:5]
            ],
            applicability_notes=applicability,
        )
        if consistency == "conflicting":
            unresolved.append(axis.axis_id)
        claims.append(claim)

    return ClaimBundle(
        claims=claims,
        question_type=qtype,
        unresolved_conflicts=unresolved,
    )


def extract_claims_from_state(state: dict[str, Any]) -> ClaimBundle:
    """Conveniencia: frame + artículos desde estado del grafo."""
    eb = state.get("evidence_bundle")
    if not isinstance(eb, dict):
        return ClaimBundle()
    arts = [a for a in (eb.get("articles") or []) if isinstance(a, dict)]
    frame = frame_from_state(state)
    qtype = frame.question_type if frame else None
    return extract_claims_deterministic(
        frame,
        arts,
        question_type=qtype,
        primary_slice=primary_slice_enabled(),
    )


def bundle_supports_claim_first(bundle: ClaimBundle | None) -> bool:
    return bool(bundle and bundle.claims)


def _contradictions_line(claim: ClinicalClaim) -> str:
    if not claim.contradicting:
        return "ninguna significativa en el pool recuperado"
    parts: list[str] = []
    for s in claim.contradicting[:4]:
        tag = s.study_type or s.evidence_role or "referencia"
        parts.append(f"PMID {s.pmid} ({tag})")
    return "; ".join(parts)


def render_claim_bundle_markdown(bundle: ClaimBundle) -> str | None:
    """Resumen claim-first (determinista + entrada LLM)."""
    if not bundle.claims:
        return None

    lines: list[str] = [
        "## Evidencia agregada por afirmaciones clínicas",
        "",
        "_Extracción determinista por eje; no sustituye guías clínicas._",
        "",
    ]
    for c in bundle.claims:
        label = c.axis_label or _axis_label_for(c.axis_id, c.direction)
        lines.append(f"### Claim: {label}")
        lines.append(f"- **Afirmación:** {c.statement}")
        lm = c.landmark_support or sorted(
            {s.landmark_acronym for s in c.support if s.landmark_acronym}
        )
        if lm:
            lines.append("- **Soporte (landmarks):** " + ", ".join(lm[:6]) + ".")
        if c.support:
            lines.append("- **Soporte (PMIDs):** " + ", ".join(s.pmid for s in c.support[:6]) + ".")
        lines.append(
            f"- **Fuerza:** {c.evidence_strength} | **Consistencia:** {c.consistency} "
            f"| **Confianza heurística:** {c.confidence:.2f}."
        )
        if c.applicability_notes:
            lines.append("- **Aplicabilidad:** " + "; ".join(c.applicability_notes) + ".")
        else:
            lines.append("- **Aplicabilidad:** según población de los estudios citados.")
        lines.append(f"- **Contradicciones:** {_contradictions_line(c)}.")
        lines.append("")

    if bundle.unresolved_conflicts:
        lines.append(
            "**Conflictos no resueltos:** "
            + ", ".join(bundle.unresolved_conflicts)
            + ". No suavizar en la redacción."
        )
    return "\n".join(lines).strip()


def claims_to_evidence_statements(bundle: ClaimBundle) -> list[dict[str, Any]]:
    """``evidence_statements`` agregados por claim (no por paper)."""
    out: list[dict[str, Any]] = []
    for c in bundle.claims:
        label = c.axis_label or c.axis_id
        stmt = (
            f"[Claim: {label}] {c.statement} "
            f"(fuerza={c.evidence_strength}, consistencia={c.consistency})."
        )
        if c.landmark_support:
            stmt += " Landmarks: " + ", ".join(c.landmark_support[:4]) + "."
        pmids = [s.pmid for s in c.support if s.pmid]
        row: dict[str, Any] = {"statement": stmt, "citation_pmids": pmids}
        if c.evidence_strength in ("high", "moderate", "low"):
            row["strength"] = c.evidence_strength
        out.append(row)
    return out
