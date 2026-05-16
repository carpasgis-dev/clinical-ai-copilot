"""
Alineación dinámica pregunta (ClinicalIntent) ↔ artículo PubMed.

Scores parciales 0–1; sin exclusión rígida. Complementa ``population_context_alignment``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent, intent_asks_cv_outcomes
from app.capabilities.evidence_rag.outcome_ontology import graded_cv_outcome_score

# Señales por concepto normalizado (título + abstract).
_CONCEPT_ARTICLE: dict[str, tuple[str, ...]] = {
    "type 2 diabetes": (
        r"\btype\s*2\s*diabet",
        r"\bt2dm\b",
        r"\bt2d\b",
        r"\bdiabetes\s*mellitus\b",
        r"\bdiabet",
    ),
    "type 1 diabetes": (
        r"\btype\s*1\s*diabet",
        r"\bt1dm\b",
        r"\bt1d\b",
        r"\binsulin[-\s]?dependent",
    ),
    "hypertension": (
        r"\bhypertens",
        r"\bhipertens",
        r"\bblood\s+pressure",
        r"\bantihypertensive",
    ),
    "older adults": (
        r"\bolder\s+adult",
        r"\belderly",
        r"\baged\s+65",
        r"\b≥\s*65",
        r"\b65\s*\+",
    ),
    "pcos": (r"\bpcos\b", r"\bpolycystic\s+ovary", r"\bpolycystic\s+ovarian"),
    "obesity": (r"\bobes", r"\boverweight", r"\bbariatric"),
    "hiv": (r"\bhiv\b", r"\baids\b", r"\bantiretrovir", r"\bhaart\b"),
    "pregnancy": (r"\bpregnan", r"\bgestation", r"\bobstetric", r"\bprenatal"),
    "sglt2": (r"\bsglt-?2", r"\bgliflozin", r"\bempagliflozin", r"\bdapagliflozin"),
    "glp1": (r"\bglp-?1", r"\bsemaglutide", r"\bliraglutide", r"\btirzepatide"),
    "metformin": (r"\bmetformin", r"\bmetformina"),
    "insulin": (r"\binsulin\b", r"\binsulina"),
    "anticoagulation": (r"\banticoagul", r"\bwarfarin", r"\bdoac\b", r"\bapixaban"),
    "cardiovascular events": (
        r"\bcardiovascular",
        r"\bcardiovasc",
        r"\bcoronary",
        r"\bheart\s+disease",
    ),
    "mace": (r"\bmace\b", r"\bmajor\s+adverse\s+cardiovascular"),
    "mortality": (r"\bmortal", r"\ball[-\s]?cause\s+death", r"\bdeath\b"),
    "heart failure": (
        r"\bheart\s+failure",
        r"\bhospitalization\s+for\s+hf",
        r"\bhf\s+hospitalization",
        r"\bhhf\b",
    ),
    "stroke": (r"\bstroke\b", r"\bcerebrovascular", r"\bictus", r"\bischemic\s+stroke"),
    "glycemic control": (r"\bhba1c\b", r"\bglycemic", r"\bglucose\s+control"),
    "placebo": (r"\bplacebo", r"\bcontrol\s+group"),
    "warfarin": (r"\bwarfarin", r"\bvitamin\s+k\s+antagonist"),
}

_YOUNG_POP = re.compile(
    r"\b(adolescents?|pediatric|paediatric|children|childhood|school-?age|infants?|teens?)\b",
    re.I,
)

# --- Capas de señal CV (outcome_score gradual, no binario) ---
_CV_GENERIC = re.compile(
    r"\b(cardiovascular|cardiovasc|coronary|heart\s+disease|atherosclerotic)\b",
    re.I,
)
_CV_ENDPOINT_MACE_DEATH = re.compile(
    r"\b(mace|major\s+adverse\s+cardiovascular|cardiovascular\s+death|cv\s+death|cardiac\s+death)\b",
    re.I,
)
_CV_ENDPOINT_HF = re.compile(
    r"\b(heart\s+failure|hospitalization\s+for\s+heart\s+failure|hhf|hf\s+hospitalization)\b",
    re.I,
)
_CV_ENDPOINT_STROKE_MI = re.compile(
    r"\b(stroke|cerebrovascular|ischemic\s+stroke|myocardial\s+infarction)\b",
    re.I,
)
_CV_ENDPOINT_MORTALITY = re.compile(
    r"\b(all[-\s]?cause\s+mortality|cardiovascular\s+mortality|cv\s+mortality)\b",
    re.I,
)

# Off-topic solo penaliza si NO hay señal CV suficiente (trials GLP-1 con peso + MACE son válidos).
_OFF_TOPIC_PSYCH = re.compile(
    r"\b(depress|depression|depressive|anxiety|psychiatric|mental\s+health)\b",
    re.I,
)
_OFF_TOPIC_WEIGHT = re.compile(
    r"\b("
    r"weight\s+loss|weight\s+reduction|body\s+weight|body\s+mass|"
    r"bmi\s+decrease|bmi\s+reduction|body\s+mass\s+index|"
    r"anthropometric|adiposity|adipose|"
    r"obesity[-\s]?related|obesity\s+endpoint|"
    r"waist\s+circumference|percent\s+weight"
    r")\b",
    re.I,
)
_OFF_TOPIC_QOL_AE = re.compile(
    r"\b(quality\s+of\s+life|qol\b|nausea|vomiting|gastrointestinal\s+adverse)\b",
    re.I,
)

_COMPARATOR_PHRASES: dict[str, tuple[str, ...]] = {
    "metformin": (
        r"\bversus\s+metformin",
        r"\bvs\.?\s+metformin",
        r"\bcompared\s+(with|to)\s+metformin",
        r"\bfrente\s+a\s+metformin",
        r"\bon\s+metformin",
        r"\bwith\s+metformin",
        r"\badd[-\s]?on\s+to\s+metformin",
        r"\bin\s+addition\s+to\s+metformin",
        r"\bbackground\s+of\s+metformin",
        r"\bmetformin\s+monotherapy",
        r"\bstandard\s+care\s+.*metformin",
        r"\bmetformin\s+alone",
    ),
    "warfarin": (
        r"\bversus\s+warfarin",
        r"\bvs\.?\s+warfarin",
        r"\bcompared\s+(with|to)\s+warfarin",
    ),
    "placebo": (
        r"\bversus\s+placebo",
        r"\bvs\.?\s+placebo",
        r"\bcompared\s+(with|to)\s+placebo",
        r"\bplacebo[-\s]?controlled",
    ),
}

_DRUG_CLASS_VS_ONLY = re.compile(
    r"\b(sglt2|sglt-2|glp-1|glp1|semaglutide|liraglutide|empagliflozin|dapagliflozin)\b"
    r".{0,80}\b(versus|vs\.?|compared\s+to)\b"
    r".{0,80}\b(sglt2|sglt-2|glp-1|glp1|semaglutide|liraglutide|empagliflozin|dapagliflozin)\b",
    re.I | re.DOTALL,
)


def _fold(s: str) -> str:
    return fold_ascii((s or "").lower())


def _concept_hit(blob: str, concept: str) -> bool:
    pats = _CONCEPT_ARTICLE.get(concept.lower())
    if not pats:
        return concept.lower() in blob
    return any(re.search(p, blob, re.I) for p in pats)


def _axis_score(requested: list[str], blob: str) -> float:
    """Promedio de coincidencias por concepto solicitado (0 si no hay petición)."""
    if not requested:
        return 0.55  # neutro: no penalizar ni inflar
    hits = sum(1 for c in requested if _concept_hit(blob, c))
    return hits / len(requested)


def _axis_score_or_alternatives(requested: list[str], blob: str) -> float:
    """
    Para intervenciones u outcomes alternativos (A o B): basta con una coincidencia fuerte.
    """
    if not requested:
        return 0.55
    hits = sum(1 for c in requested if _concept_hit(blob, c))
    if hits == 0:
        return 0.08
    return min(1.0, 0.72 + 0.14 * hits)


@dataclass(frozen=True, slots=True)
class AlignmentScores:
    population_score: float
    intervention_score: float
    outcome_score: float
    comparator_score: float
    study_type_score: float
    age_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "population_score": round(self.population_score, 3),
            "intervention_score": round(self.intervention_score, 3),
            "outcome_score": round(self.outcome_score, 3),
            "comparator_score": round(self.comparator_score, 3),
            "study_type_score": round(self.study_type_score, 3),
            "age_score": round(self.age_score, 3),
        }


def _age_score(intent: ClinicalIntent, blob: str) -> float:
    if intent.age_min is None and intent.age_max is None:
        return 0.6
    score = 0.75
    young = bool(_YOUNG_POP.search(blob))
    if intent.age_min is not None and intent.age_min >= 65:
        if young and not re.search(r"\bolder\s+adult|\belderly|\baged\b", blob, re.I):
            return 0.15
        if re.search(r"\bolder\s+adult|\belderly|\baged\b", blob, re.I):
            return 0.95
        return 0.55
    if intent.age_max is not None and intent.age_max <= 18:
        if young:
            return 0.9
        return 0.25
    return score


def _population_score(intent: ClinicalIntent, blob: str) -> float:
    base = _axis_score(intent.population, blob)
    if not intent.population:
        return base

    # Penalización suave por ruido no invitado (PCOS, T1DM, etc.)
    noise_pen = 0.0
    for noise in intent.population_noise:
        if _concept_hit(blob, noise):
            # Si el artículo también cumple población pedida, atenuar penalización
            if base >= 0.5:
                noise_pen += 0.12
            else:
                noise_pen += 0.45
    # Conflicto T2 pedido + T1 fuerte en artículo
    if "type 2 diabetes" in [p.lower() for p in intent.population]:
        if _concept_hit(blob, "type 1 diabetes") and not _concept_hit(blob, "type 2 diabetes"):
            noise_pen = max(noise_pen, 0.55)
    return max(0.05, min(1.0, base - min(0.75, noise_pen)))


def _cv_endpoint_families_hit(blob: str) -> int:
    """Número de familias de endpoint CV distintas (0–4)."""
    n = 0
    if _CV_ENDPOINT_MACE_DEATH.search(blob):
        n += 1
    if _CV_ENDPOINT_HF.search(blob):
        n += 1
    if _CV_ENDPOINT_STROKE_MI.search(blob):
        n += 1
    if _CV_ENDPOINT_MORTALITY.search(blob):
        n += 1
    return n


def _cv_outcome_graded_score(blob: str) -> float:
    """
    Score gradual de desenlace CV (tier lexical).

    | Señal                         | ~score |
    |-------------------------------|--------|
    | solo «cardiovascular» genérico | 0.40   |
    | 1 familia endpoint (p. ej. MACE) | 0.72 |
    | MACE + CV death               | 0.85   |
    | MACE + HF + stroke (+/- mort.) | 0.95–1.0 |
    """
    families = _cv_endpoint_families_hit(blob)
    has_mace_death = bool(_CV_ENDPOINT_MACE_DEATH.search(blob))
    has_generic = bool(_CV_GENERIC.search(blob))

    ontology_score = graded_cv_outcome_score(blob)
    if families >= 3:
        return max(0.98, ontology_score)
    if families == 2:
        lexical = 0.88 if has_mace_death else 0.82
        return max(lexical, ontology_score)
    if families == 1:
        lexical = 0.72 if has_mace_death else 0.65
        return max(lexical, ontology_score)
    if has_generic:
        return max(0.40, ontology_score)
    return ontology_score


def _cv_outcome_signal(blob: str) -> bool:
    """True si hay señal CV mínima útil (no solo genérico vago)."""
    return _cv_outcome_graded_score(blob) >= 0.55


def _off_topic_outcome_penalty(intent: ClinicalIntent, blob: str) -> float:
    """
    Penalización solo sin señal CV clara: peso/depresión coexisten con MACE → sin penalizar.
    """
    if not intent_asks_cv_outcomes(intent):
        return 0.0
    if _cv_outcome_graded_score(blob) >= 0.55:
        return 0.0

    pen = 0.0
    if _OFF_TOPIC_PSYCH.search(blob):
        pen += 0.25
    if _OFF_TOPIC_WEIGHT.search(blob):
        pen += 0.20
    if _OFF_TOPIC_QOL_AE.search(blob):
        pen += 0.12
    return min(0.40, pen)


def _outcome_score(intent: ClinicalIntent, blob: str) -> float:
    if not intent.outcomes:
        return 0.55

    if intent_asks_cv_outcomes(intent):
        graded = _cv_outcome_graded_score(blob)
        if graded > 0.0:
            base = graded
            concept_hits = sum(1 for o in intent.outcomes if _concept_hit(blob, o))
            if concept_hits:
                base = min(1.0, base + 0.04 * min(concept_hits, 3))
        else:
            base = _axis_score_or_alternatives(intent.outcomes, blob) * 0.50

        base -= _off_topic_outcome_penalty(intent, blob)

        # Depresión como outcome principal sin CV: techo bajo (no acumular con penalización).
        if _OFF_TOPIC_PSYCH.search(blob) and graded < 0.55:
            base = min(base, 0.22)

        return max(0.05, min(1.0, base))

    return _axis_score_or_alternatives(intent.outcomes, blob)


def _comparator_score(intent: ClinicalIntent, blob: str) -> float:
    if not intent.comparator:
        return 0.6

    per_comp: list[float] = []
    for comp in intent.comparator:
        key = comp.lower()
        phrases = _COMPARATOR_PHRASES.get(key, (rf"\b{re.escape(key)}\b",))
        phrase_hit = any(re.search(p, blob, re.I) for p in phrases)
        word_hit = _concept_hit(blob, key)

        if phrase_hit:
            per_comp.append(0.95)
        elif word_hit and key == "metformin":
            per_comp.append(0.38)
        elif word_hit:
            per_comp.append(0.48)
        else:
            per_comp.append(0.1)

    base = max(per_comp) if per_comp else 0.1

    # Pregunta «frente a metformina»: comparar solo clases (SGLT2 vs GLP-1) no cuenta.
    if "metformin" in [c.lower() for c in intent.comparator]:
        met_phrase = any(
            re.search(p, blob, re.I) for p in _COMPARATOR_PHRASES.get("metformin", ())
        )
        if _DRUG_CLASS_VS_ONLY.search(blob) and not met_phrase:
            base = min(base, 0.22)

    return max(0.05, min(1.0, base))


def score_paper_alignment(
    intent: ClinicalIntent | Mapping[str, object] | None,
    title: str,
    abstract_snippet: str = "",
) -> AlignmentScores:
    """Scores parciales 0–1 para un artículo frente a la intención clínica."""
    if intent is None:
        return AlignmentScores(0.55, 0.55, 0.55, 0.55, 0.55, 0.55)
    if not isinstance(intent, ClinicalIntent):
        intent = ClinicalIntent.from_dict(intent)  # type: ignore[arg-type]

    from app.capabilities.evidence_rag.evidence_rerank import (
        infer_study_type_from_title,
        study_design_weight,
    )

    blob = _fold(f"{title} {abstract_snippet}")
    st = infer_study_type_from_title(title)
    st_score = study_design_weight(st, clinical_intent=intent)

    pop = _population_score(intent, blob)
    iv = _axis_score_or_alternatives(intent.interventions, blob)
    out = _outcome_score(intent, blob)
    comp = _comparator_score(intent, blob)
    age = _age_score(intent, blob)

    return AlignmentScores(
        population_score=pop,
        intervention_score=iv,
        outcome_score=out,
        comparator_score=comp,
        study_type_score=st_score,
        age_score=age,
    )


def alignment_composite(
    scores: AlignmentScores,
    *,
    priority_axis: str = "intervention",
    intent: ClinicalIntent | None = None,
) -> float:
    """
    Puntuación 0–1 para re-ranking; pesos según eje prioritario de la pregunta.
    """
    axis = (priority_axis or "intervention").strip().lower()
    if axis == "outcome":
        w_pop, w_iv, w_out, w_comp = 0.14, 0.12, 0.42, 0.12
    elif axis == "population":
        w_pop, w_iv, w_out, w_comp = 0.36, 0.14, 0.20, 0.08
    else:
        w_pop, w_iv, w_out, w_comp = 0.22, 0.28, 0.28, 0.10

    # Preguntas con desenlaces CV/MACE: subir peso de outcome aunque el eje prioritario sea intervención.
    if intent and intent_asks_cv_outcomes(intent) and axis != "outcome":
        w_out = max(w_out, 0.36)
        w_iv = min(w_iv, 0.24)

    # Pregunta con comparador explícito (p. ej. vs metformina): más peso al comparador.
    if intent and intent.comparator:
        w_comp = max(w_comp, 0.14)
        w_out = max(w_out, 0.24)

    raw = (
        w_pop * scores.population_score
        + w_iv * scores.intervention_score
        + w_out * scores.outcome_score
        + w_comp * scores.comparator_score
        + 0.10 * scores.study_type_score
        + 0.06 * scores.age_score
    )

    # Outcome-centric: no dejar pasar «solo GLP-1 + depresión» con outcome_score bajo.
    if intent and intent_asks_cv_outcomes(intent):
        if scores.intervention_score >= 0.7 and scores.outcome_score < 0.3:
            raw *= 0.68
        if scores.outcome_score < 0.22:
            raw = min(raw, 0.45)

    return max(0.05, min(1.0, raw))
