"""
Stage 1 — descomposición clínica estable (Clinical Intent Graph).

Separa la pregunta NL en objetivo clínico, PICO enriquecido, política de evidencia
y ensayos landmark esperados. Alimenta retrieval, filtrado pre-rerank y calibración.

No sustituye ``ClinicalIntent`` (compatibilidad downstream); lo extiende vía ``to_clinical_intent()``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Union

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.capabilities.evidence_rag.clinical_intent import (
    ClinicalIntent,
    extract_clinical_intent,
    infer_priority_axis,
    primary_outcome_theme,
)
from app.schemas.copilot_state import ClinicalContext

QuestionType = Literal[
    "comparative_effectiveness",
    "treatment_efficacy",
    "treatment_selection",
    "adverse_effect",
    "prognosis",
    "diagnosis",
    "mechanism",
    "guideline_lookup",
    "general",
]

EvidencePreference = Literal[
    "guideline",
    "meta_analysis",
    "RCT",
    "large_observational",
    "systematic_review",
    "cohort",
    "case_series",
    "basic_science",
]

SuppressEvidence = Literal[
    "preclinical",
    "mechanistic",
    "prediction_model",
    "epidemiology",
    "ehealth",
    "editorial",
    "case_report",
]

_DOAC_AGENTS = (
    "DOAC",
    "NOAC",
    "direct oral anticoagulant",
    "apixaban",
    "rivaroxaban",
    "dabigatran",
    "edoxaban",
)


@dataclass
class ClinicalIntentGraph:
    """
    Representación intermedia estable entre NL y retrieval/síntesis.

    ``question_type`` es el nodo principal; PICO es derivado y enriquecido.
    """

    question_type: str = "general"
    population: list[str] = field(default_factory=list)
    intervention: list[str] = field(default_factory=list)
    comparator: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    preferred_evidence: list[str] = field(default_factory=list)
    suppress_evidence: list[str] = field(default_factory=list)
    expected_landmark_trials: list[str] = field(default_factory=list)
    priority_axis: str = "intervention"
    outcome_theme: str = "general"
    age_min: int | None = None
    age_max: int | None = None
    population_noise: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "population": list(self.population),
            "intervention": list(self.intervention),
            "comparator": list(self.comparator),
            "outcomes": list(self.outcomes),
            "preferred_evidence": list(self.preferred_evidence),
            "suppress_evidence": list(self.suppress_evidence),
            "expected_landmark_trials": list(self.expected_landmark_trials),
            "priority_axis": self.priority_axis,
            "outcome_theme": self.outcome_theme,
            "age_min": self.age_min,
            "age_max": self.age_max,
            "population_noise": list(self.population_noise),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> ClinicalIntentGraph:
        if not raw:
            return cls()
        return cls(
            question_type=str(raw.get("question_type") or "general"),
            population=[str(x) for x in (raw.get("population") or []) if str(x).strip()],
            intervention=[str(x) for x in (raw.get("intervention") or []) if str(x).strip()],
            comparator=[str(x) for x in (raw.get("comparator") or []) if str(x).strip()],
            outcomes=[str(x) for x in (raw.get("outcomes") or []) if str(x).strip()],
            preferred_evidence=[
                str(x) for x in (raw.get("preferred_evidence") or []) if str(x).strip()
            ],
            suppress_evidence=[
                str(x) for x in (raw.get("suppress_evidence") or []) if str(x).strip()
            ],
            expected_landmark_trials=[
                str(x)
                for x in (raw.get("expected_landmark_trials") or [])
                if str(x).strip()
            ],
            priority_axis=str(raw.get("priority_axis") or "intervention"),
            outcome_theme=str(raw.get("outcome_theme") or "general"),
            age_min=_safe_int(raw.get("age_min")),
            age_max=_safe_int(raw.get("age_max")),
            population_noise=[
                str(x) for x in (raw.get("population_noise") or []) if str(x).strip()
            ],
        )

    def to_clinical_intent(self) -> ClinicalIntent:
        """Puente hacia rerank / alignment existentes."""
        return ClinicalIntent(
            population=list(self.population),
            interventions=list(self.intervention),
            comparator=list(self.comparator),
            outcomes=list(self.outcomes),
            evidence_preference=_preferred_to_legacy(self.preferred_evidence),
            age_min=self.age_min,
            age_max=self.age_max,
            population_noise=list(self.population_noise),
            priority_axis=self.priority_axis,  # type: ignore[arg-type]
            question_type=self.question_type,
        )


def _safe_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _preferred_to_legacy(preferred: list[str]) -> list[str]:
    mapping = {
        "meta_analysis": "meta-analysis",
        "RCT": "rct",
        "guideline": "guideline",
        "systematic_review": "meta-analysis",
    }
    return _dedupe([mapping.get(p, p) for p in preferred])


def classify_question_type(query: str, intent: ClinicalIntent | None) -> str:
    """Clasificación del objetivo clínico (sin LLM)."""
    fl = fold_ascii(query or "")
    if not fl:
        return "general"

    if re.search(r"\b(guia|guía|guideline|consensus|recomendacion|recomendación)\b", fl):
        return "guideline_lookup"
    if re.search(r"\b(mecanismo|pathophysiology|pathogenesis|molecular|how does)\b", fl):
        return "mechanism"
    if re.search(r"\b(prognos|supervivencia|survival|curso natural|riesgo a \d)\b", fl):
        return "prognosis"
    if re.search(r"\b(diagnos|sensibilidad|especificidad|accuracy|screening test)\b", fl):
        return "diagnosis"
    if re.search(r"\b(seguridad|adverse|efectos adversos|toxicidad|farmacovigil)\b", fl):
        return "adverse_effect"
    if re.search(
        r"\b(mejor opcion|elegir|seleccion|selección|primera linea|primera línea|"
        r"que terapia|que fármaco|que farmaco)\b",
        fl,
    ):
        return "treatment_selection"

    if re.search(
        r"\b(doac|noac|apixaban|rivaroxaban|dabigatran|edoxaban|anticoagul)\b",
        fl,
    ) and re.search(r"\b(warfarin|warfarina|avk)\b", fl):
        return "comparative_effectiveness"

    if intent and intent.comparator and intent.interventions:
        return "comparative_effectiveness"
    if re.search(r"\b(frente a|versus|vs\.?|compar)\b", fl):
        if re.search(
            r"\b(doac|noac|warfarin|anticoagul|sglt2|glp|metformin|apixaban|rivaroxaban)\b",
            fl,
        ):
            return "comparative_effectiveness"
    if re.search(
        r"\b(que evidencia|reducir|prevencion|prevención|mace|riesgo cardiovascular|"
        r"outcome|eficacia|beneficio|stroke prevention)\b",
        fl,
    ):
        return "treatment_efficacy"
    return "general"


def infer_question_type(query: str, intent: ClinicalIntent | None) -> str:
    """Alias estable para módulos que ya importaban desde clinical_answerability."""
    return classify_question_type(query, intent)


def _enrich_population(fl: str, intent: ClinicalIntent) -> list[str]:
    pop = list(intent.population)
    if re.search(
        r"\b(fibrilacion auricular|fibrilación auricular|atrial fibrill|fa\b|nvaf)\b",
        fl,
    ):
        if not any("fibrill" in p.lower() or p.lower() == "af" for p in pop):
            pop.append("atrial fibrillation")
    if re.search(r"\b(diabet|dm2|t2dm)\b", fl) and not any("diabet" in p.lower() for p in pop):
        pop.append("type 2 diabetes")
    return _dedupe(pop)


def _enrich_interventions(fl: str, intent: ClinicalIntent) -> list[str]:
    iv = list(intent.interventions)
    blob = " ".join(iv).lower()
    if re.search(r"\b(doac|noac|anticoagul|apixaban|rivaroxaban|dabigatran|edoxaban)\b", fl):
        for agent in _DOAC_AGENTS:
            if agent.lower() not in blob:
                iv.append(agent)
                blob += f" {agent.lower()}"
        if "anticoagulation" not in blob:
            iv.append("anticoagulation")
    return _dedupe(iv)


def _enrich_comparator(fl: str, intent: ClinicalIntent) -> list[str]:
    comp = list(intent.comparator)
    if re.search(r"\b(warfarin|avk|vitamin k|antagonista)\b", fl):
        if not any("warfarin" in c.lower() for c in comp):
            comp.append("warfarin")
        if not any("vitamin k" in c.lower() for c in comp):
            comp.append("vitamin K antagonist")
    if re.search(r"\b(doac|noac)\b", fl) and re.search(r"\b(frente a|versus|vs\.?|compar)\b", fl):
        if not any("warfarin" in c.lower() for c in comp):
            comp.append("warfarin")
    return _dedupe(comp)


def _enrich_outcomes(fl: str, intent: ClinicalIntent) -> list[str]:
    out = list(intent.outcomes)
    if re.search(r"\b(ictus|stroke|emboli|embolism)\b", fl):
        if not any("stroke" in o.lower() for o in out):
            out.append("stroke")
        if not any("embol" in o.lower() for o in out):
            out.append("systemic embolism")
    if re.search(r"\b(hemorrag|bleeding|sangrado)\b", fl):
        for label in ("major bleeding", "intracranial hemorrhage"):
            if not any(label in o.lower() for o in out):
                out.append(label)
    return _dedupe(out)


def _is_anticoag_comparative(graph: ClinicalIntentGraph) -> bool:
    iv_blob = fold_ascii(" ".join(graph.intervention))
    comp_blob = fold_ascii(" ".join(graph.comparator))
    return (
        graph.question_type in ("comparative_effectiveness", "treatment_selection")
        and (
            "warfarin" in comp_blob
            or "vitamin k" in comp_blob
        )
        and any(
            x in iv_blob
            for x in ("doac", "noac", "apixaban", "rivaroxaban", "dabigatran", "edoxaban", "anticoagul")
        )
    )


def infer_expected_landmark_trials(graph: ClinicalIntentGraph) -> list[str]:
    """Ensayos landmark esperados según comparación terapéutica."""
    if _is_anticoag_comparative(graph):
        from app.capabilities.evidence_rag.landmark_registry import expected_acronyms_for_anticoag

        return list(expected_acronyms_for_anticoag())
    intent = graph.to_clinical_intent()
    theme = primary_outcome_theme(intent)
    if theme == "cv" and any(
        x in fold_ascii(" ".join(graph.intervention))
        for x in ("sglt2", "glp", "empagliflozin", "semaglutide", "dapagliflozin")
    ):
        from app.capabilities.evidence_rag.landmark_registry import LANDMARK_CVOTS

        return [t.acronym for t in LANDMARK_CVOTS[:8]]
    return []


def build_clinical_intent_graph(
    query: str,
    clinical_context: Optional[Union[ClinicalContext, dict, Mapping[str, Any]]] = None,
) -> ClinicalIntentGraph:
    """
    Stage 1: NL + cohorte → grafo clínico estable para políticas downstream.
    """
    q = (query or "").strip()
    fl = fold_ascii(q)
    intent = extract_clinical_intent(q, clinical_context)
    qtype = classify_question_type(q, intent)
    from app.capabilities.evidence_rag.clinical_semantics import policy_for_question_type

    policy = policy_for_question_type(qtype)

    graph = ClinicalIntentGraph(
        question_type=qtype,
        population=_enrich_population(fl, intent),
        intervention=_enrich_interventions(fl, intent),
        comparator=_enrich_comparator(fl, intent),
        outcomes=_enrich_outcomes(fl, intent),
        preferred_evidence=list(policy.preferred_evidence),
        suppress_evidence=list(policy.suppress_evidence),
        priority_axis=intent.priority_axis,
        outcome_theme=primary_outcome_theme(intent),
        age_min=intent.age_min,
        age_max=intent.age_max,
        population_noise=list(intent.population_noise),
    )
    graph.expected_landmark_trials = infer_expected_landmark_trials(graph)
    if not graph.priority_axis:
        graph.priority_axis = infer_priority_axis(q, graph.to_clinical_intent())
    intent.question_type = qtype
    return graph


def graph_from_state(state: Mapping[str, Any]) -> ClinicalIntentGraph | None:
    raw = state.get("clinical_intent_graph")
    if isinstance(raw, ClinicalIntentGraph):
        return raw
    if isinstance(raw, dict):
        return ClinicalIntentGraph.from_dict(raw)
    raw_intent = state.get("clinical_intent")
    if isinstance(raw_intent, dict):
        g = ClinicalIntentGraph.from_dict(
            {
                "question_type": raw_intent.get("question_type"),
                "population": raw_intent.get("population"),
                "intervention": raw_intent.get("interventions"),
                "comparator": raw_intent.get("comparator"),
                "outcomes": raw_intent.get("outcomes"),
                "preferred_evidence": raw_intent.get("evidence_preference"),
                "priority_axis": raw_intent.get("priority_axis"),
                "age_min": raw_intent.get("age_min"),
                "age_max": raw_intent.get("age_max"),
                "population_noise": raw_intent.get("population_noise"),
            }
        )
        g.outcome_theme = primary_outcome_theme(g.to_clinical_intent())
        return g
    return None


def landmarks_found_in_articles(
    articles: list[dict[str, Any]],
    expected: list[str],
) -> list[str]:
    """Acrónimos landmark detectados en títulos/abstracts."""
    if not expected or not articles:
        return []
    found: list[str] = []
    for art in articles:
        blob = fold_ascii(f"{art.get('title', '')} {art.get('abstract_snippet', '')}")
        for lm in expected:
            key = fold_ascii(lm).replace(" ", "")
            if key in blob.replace(" ", "") and lm not in found:
                found.append(lm)
    return found
