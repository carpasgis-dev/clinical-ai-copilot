"""
Esquema semántico canónico del copiloto clínico (single source of truth).

``ClinicalConcept`` — ontología curada (intervención, desenlace, población, supresión).
``ClinicalEvidenceFrame`` — estado operativo compartido por retrieval, policy, rerank,
calibración, agregación y (futuro) claims.

Los módulos periféricos (``lexical_expansion``, ``landmark_registry``, ``evidence_policy``)
deben **leer** de aquí, no duplicar listas de fármacos ni políticas.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from app.capabilities.clinical_sql.terminology import fold_ascii

ConceptKind = Literal[
    "intervention",
    "outcome",
    "population",
    "comparator",
    "suppress_category",
    "evidence_design",
]

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


@dataclass(frozen=True, slots=True)
class ClinicalConcept:
    """
    Entidad clínica curada: una sola definición para expansión PubMed, landmarks y agregación.
    """

    id: str
    label_es: str
    kind: ConceptKind
    aliases: tuple[str, ...] = ()
    lexical_tokens: tuple[str, ...] = ()
    drug_members: tuple[str, ...] = ()
    pubmed_class_clause: str | None = None
    landmark_acronyms: tuple[str, ...] = ()
    match_pattern: str | None = None

    def matches_text(self, blob: str) -> bool:
        if self.match_pattern:
            return bool(re.search(self.match_pattern, blob, re.I))
        fl = fold_ascii(blob)
        for alias in (self.id, *self.aliases, *self.drug_members):
            a = fold_ascii(alias)
            if len(a) >= 3 and a in fl:
                return True
        return False


@dataclass(frozen=True, slots=True)
class QuestionTypePolicy:
    """Política de evidencia por tipo de pregunta."""

    question_type: str
    preferred_evidence: tuple[str, ...]
    suppress_evidence: tuple[str, ...]
    therapeutic_objective: bool = False


# ---------------------------------------------------------------------------
# Registro canónico (extender aquí, no en mesh_lite / clinical_concepts / graph)
# ---------------------------------------------------------------------------

CLINICAL_CONCEPTS: dict[str, ClinicalConcept] = {
    "sglt2_class": ClinicalConcept(
        id="sglt2_class",
        label_es="Inhibidores SGLT2",
        kind="intervention",
        aliases=("sglt2", "gliflozin", "iSGLT2"),
        lexical_tokens=("sglt2",),
        drug_members=("empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin"),
        landmark_acronyms=("EMPA-REG OUTCOME", "DECLARE-TIMI 58", "CANVAS", "CREDENCE"),
        match_pattern=r"\b(sglt2|gliflozin|empagliflozin|dapagliflozin|canagliflozin)\b",
    ),
    "glp1_ra": ClinicalConcept(
        id="glp1_ra",
        label_es="Agonistas GLP-1 / incretinas",
        kind="intervention",
        aliases=("glp-1", "glp1", "incretin"),
        lexical_tokens=("glp1",),
        drug_members=("semaglutide", "liraglutide", "dulaglutide", "tirzepatide"),
        landmark_acronyms=("LEADER", "SUSTAIN-6", "REWIND", "SELECT"),
        match_pattern=r"\b(glp-?1|semaglutide|liraglutide|dulaglutide|tirzepatide)\b",
    ),
    "metformin": ClinicalConcept(
        id="metformin",
        label_es="Metformina",
        kind="intervention",
        aliases=("metformina",),
        lexical_tokens=("metformin", "metform"),
        match_pattern=r"\b(metformin|metformina)\b",
    ),
    "doac_class": ClinicalConcept(
        id="doac_class",
        label_es="Anticoagulantes orales directos (DOAC)",
        kind="intervention",
        aliases=("doac", "noac", "anticoagulation", "direct oral anticoagulant"),
        drug_members=("apixaban", "rivaroxaban", "dabigatran", "edoxaban"),
        pubmed_class_clause=(
            "(\"anticoagulants\"[Mesh] OR \"direct oral anticoagulant\"[tiab] "
            "OR DOAC[tiab] OR NOAC[tiab])"
        ),
        landmark_acronyms=("ARISTOTLE", "RE-LY", "ROCKET AF", "ENGAGE AF"),
        match_pattern=(
            r"\b(doac|noac|apixaban|rivaroxaban|dabigatran|edoxaban|anticoagul)\b"
        ),
    ),
    "warfarin": ClinicalConcept(
        id="warfarin",
        label_es="Warfarina / antagonistas vitamina K",
        kind="comparator",
        aliases=("warfarina", "vitamin k antagonist", "avk"),
        lexical_tokens=("warfarin",),
        match_pattern=r"\b(warfarin|warfarina|vitamin k antagonist)\b",
    ),
    "outcome_mace": ClinicalConcept(
        id="outcome_mace",
        label_es="MACE / eventos cardiovasculares mayores",
        kind="outcome",
        aliases=("mace", "cardiovascular events", "major adverse cardiovascular"),
        match_pattern=r"\b(mace|major adverse cardiovascular)\b",
    ),
    "outcome_hf_hosp": ClinicalConcept(
        id="outcome_hf_hosp",
        label_es="Hospitalización por insuficiencia cardíaca",
        kind="outcome",
        aliases=("heart failure hospitalization",),
        match_pattern=r"\b(heart failure hospitalization|hospitalization for heart failure)\b",
    ),
    "outcome_stroke": ClinicalConcept(
        id="outcome_stroke",
        label_es="Ictus / embolia sistémica",
        kind="outcome",
        aliases=("stroke", "systemic embolism"),
        lexical_tokens=("stroke",),
        match_pattern=r"\b(stroke|systemic embolism|embolic stroke)\b",
    ),
    "outcome_bleeding": ClinicalConcept(
        id="outcome_bleeding",
        label_es="Hemorragia mayor / intracraneal",
        kind="outcome",
        aliases=("major bleeding", "intracranial hemorrhage"),
        match_pattern=r"\b(major bleeding|intracranial hemorrhage)\b",
    ),
    "pop_t2dm": ClinicalConcept(
        id="pop_t2dm",
        label_es="Diabetes mellitus tipo 2",
        kind="population",
        aliases=("type 2 diabetes", "t2dm", "diabetes tipo 2"),
        lexical_tokens=("diabet", "diabetes"),
        match_pattern=r"\b(type 2 diabetes|t2dm|diabetes mellitus)\b",
    ),
    "pop_af": ClinicalConcept(
        id="pop_af",
        label_es="Fibrilación auricular",
        kind="population",
        aliases=("atrial fibrillation", "nvaf", "fa"),
        lexical_tokens=("fibrillat", "atrial fibrillation"),
        match_pattern=r"\b(atrial fibrillation|nvaf)\b",
    ),
    "pop_older_adults": ClinicalConcept(
        id="pop_older_adults",
        label_es="Adultos mayores (≥65)",
        kind="population",
        aliases=("older adults", "elderly"),
        match_pattern=r"\b(older adult|elderly|aged 65)\b",
    ),
    "suppress_topical_report": ClinicalConcept(
        id="suppress_topical_report",
        label_es="Informe contextual / summit (no terapéutico directo)",
        kind="suppress_category",
        aliases=("summit report", "conference report", "year in review"),
        match_pattern=(
            r"\b(summit report|conference report|workshop report|year in review|"
            r"state of the science|expert panel report)\b"
        ),
    ),
    "suppress_ehealth": ClinicalConcept(
        id="suppress_ehealth",
        label_es="eHealth / telemedicina (ruido)",
        kind="suppress_category",
        aliases=("telemedicine", "e-health", "digital health"),
        match_pattern=r"\b(telemedicine|e-?health|digital health)\b",
    ),
    "suppress_prediction_model": ClinicalConcept(
        id="suppress_prediction_model",
        label_es="Modelo predictivo / ML (ruido)",
        kind="suppress_category",
        aliases=("machine learning", "prediction model"),
        match_pattern=r"\b(machine learning|prediction model|deep learning)\b",
    ),
}

QUESTION_TYPE_POLICIES: dict[str, QuestionTypePolicy] = {
    "comparative_effectiveness": QuestionTypePolicy(
        question_type="comparative_effectiveness",
        preferred_evidence=("guideline", "meta_analysis", "RCT", "large_observational"),
        suppress_evidence=(
            "preclinical",
            "mechanistic",
            "prediction_model",
            "epidemiology",
            "ehealth",
            "topical_report",
        ),
        therapeutic_objective=True,
    ),
    "treatment_efficacy": QuestionTypePolicy(
        question_type="treatment_efficacy",
        preferred_evidence=("RCT", "meta_analysis", "guideline", "systematic_review"),
        suppress_evidence=(
            "preclinical",
            "mechanistic",
            "prediction_model",
            "ehealth",
            "topical_report",
        ),
        therapeutic_objective=True,
    ),
    "treatment_selection": QuestionTypePolicy(
        question_type="treatment_selection",
        preferred_evidence=("guideline", "meta_analysis", "RCT", "large_observational"),
        suppress_evidence=(
            "preclinical",
            "mechanistic",
            "prediction_model",
            "epidemiology",
            "ehealth",
            "topical_report",
        ),
        therapeutic_objective=True,
    ),
    "adverse_effect": QuestionTypePolicy(
        question_type="adverse_effect",
        preferred_evidence=("meta_analysis", "RCT", "cohort", "large_observational"),
        suppress_evidence=("mechanistic", "prediction_model", "ehealth", "epidemiology"),
        therapeutic_objective=False,
    ),
    "prognosis": QuestionTypePolicy(
        question_type="prognosis",
        preferred_evidence=("cohort", "large_observational", "meta_analysis"),
        suppress_evidence=("preclinical", "mechanistic", "prediction_model", "ehealth"),
        therapeutic_objective=False,
    ),
    "diagnosis": QuestionTypePolicy(
        question_type="diagnosis",
        preferred_evidence=("RCT", "cohort", "meta_analysis"),
        suppress_evidence=("preclinical", "mechanistic", "ehealth"),
        therapeutic_objective=False,
    ),
    "mechanism": QuestionTypePolicy(
        question_type="mechanism",
        preferred_evidence=("systematic_review", "basic_science"),
        suppress_evidence=("prediction_model", "ehealth", "epidemiology"),
        therapeutic_objective=False,
    ),
    "guideline_lookup": QuestionTypePolicy(
        question_type="guideline_lookup",
        preferred_evidence=("guideline", "meta_analysis", "systematic_review"),
        suppress_evidence=("preclinical", "prediction_model", "ehealth"),
        therapeutic_objective=False,
    ),
    "general": QuestionTypePolicy(
        question_type="general",
        preferred_evidence=("meta_analysis", "RCT", "systematic_review"),
        suppress_evidence=("prediction_model", "ehealth"),
        therapeutic_objective=False,
    ),
}

_ALIAS_INDEX: dict[str, str] = {}


def _build_alias_index() -> None:
    global _ALIAS_INDEX
    if _ALIAS_INDEX:
        return
    for cid, concept in CLINICAL_CONCEPTS.items():
        _ALIAS_INDEX[fold_ascii(cid)] = cid
        for alias in concept.aliases:
            _ALIAS_INDEX[fold_ascii(alias)] = cid
        for drug in concept.drug_members:
            _ALIAS_INDEX[fold_ascii(drug)] = cid
        for tok in concept.lexical_tokens:
            _ALIAS_INDEX[fold_ascii(tok)] = cid


_build_alias_index()


def get_concept(concept_id: str) -> ClinicalConcept | None:
    return CLINICAL_CONCEPTS.get(concept_id)


def resolve_concept_id(token: str) -> str | None:
    """Resuelve alias / fármaco / id → concept_id canónico."""
    key = fold_ascii((token or "").strip())
    if not key:
        return None
    return _ALIAS_INDEX.get(key)


def concepts_by_kind(kind: ConceptKind) -> list[ClinicalConcept]:
    return [c for c in CLINICAL_CONCEPTS.values() if c.kind == kind]


def intervention_concepts() -> list[ClinicalConcept]:
    return concepts_by_kind("intervention")


def policy_for_question_type(question_type: str | None) -> QuestionTypePolicy:
    q = question_type or "general"
    return QUESTION_TYPE_POLICIES.get(q, QUESTION_TYPE_POLICIES["general"])


def expected_landmarks_for_concepts(concept_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cid in concept_ids:
        c = get_concept(cid)
        if not c:
            continue
        for ac in c.landmark_acronyms:
            if ac not in seen:
                seen.add(ac)
                out.append(ac)
    return out


def classify_intervention_in_text(blob: str) -> list[str]:
    """concept_ids de intervención detectados en título/abstract."""
    found: list[str] = []
    for c in intervention_concepts():
        if c.matches_text(blob):
            found.append(c.id)
    return found


@dataclass
class ClinicalEvidenceFrame:
    """
    Marco de evidencia canónico — todo el pipeline opera sobre esta instancia.
    """

    question_type: str = "general"
    population: list[str] = field(default_factory=list)
    intervention: list[str] = field(default_factory=list)
    comparator: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    population_concept_ids: list[str] = field(default_factory=list)
    intervention_concept_ids: list[str] = field(default_factory=list)
    comparator_concept_ids: list[str] = field(default_factory=list)
    outcome_concept_ids: list[str] = field(default_factory=list)
    preferred_evidence: list[str] = field(default_factory=list)
    suppress_evidence: list[str] = field(default_factory=list)
    expected_landmark_trials: list[str] = field(default_factory=list)
    outcome_theme: str = "general"
    priority_axis: str = "intervention"
    therapeutic_objective: bool = False
    age_min: int | None = None
    age_max: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "population": list(self.population),
            "intervention": list(self.intervention),
            "comparator": list(self.comparator),
            "outcomes": list(self.outcomes),
            "population_concept_ids": list(self.population_concept_ids),
            "intervention_concept_ids": list(self.intervention_concept_ids),
            "comparator_concept_ids": list(self.comparator_concept_ids),
            "outcome_concept_ids": list(self.outcome_concept_ids),
            "preferred_evidence": list(self.preferred_evidence),
            "suppress_evidence": list(self.suppress_evidence),
            "expected_landmark_trials": list(self.expected_landmark_trials),
            "outcome_theme": self.outcome_theme,
            "priority_axis": self.priority_axis,
            "therapeutic_objective": self.therapeutic_objective,
            "age_min": self.age_min,
            "age_max": self.age_max,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> ClinicalEvidenceFrame:
        if not raw:
            return cls()
        return cls(
            question_type=str(raw.get("question_type") or "general"),
            population=[str(x) for x in (raw.get("population") or [])],
            intervention=[str(x) for x in (raw.get("intervention") or [])],
            comparator=[str(x) for x in (raw.get("comparator") or [])],
            outcomes=[str(x) for x in (raw.get("outcomes") or [])],
            population_concept_ids=[
                str(x) for x in (raw.get("population_concept_ids") or [])
            ],
            intervention_concept_ids=[
                str(x) for x in (raw.get("intervention_concept_ids") or [])
            ],
            comparator_concept_ids=[
                str(x) for x in (raw.get("comparator_concept_ids") or [])
            ],
            outcome_concept_ids=[str(x) for x in (raw.get("outcome_concept_ids") or [])],
            preferred_evidence=[str(x) for x in (raw.get("preferred_evidence") or [])],
            suppress_evidence=[str(x) for x in (raw.get("suppress_evidence") or [])],
            expected_landmark_trials=[
                str(x) for x in (raw.get("expected_landmark_trials") or [])
            ],
            outcome_theme=str(raw.get("outcome_theme") or "general"),
            priority_axis=str(raw.get("priority_axis") or "intervention"),
            therapeutic_objective=bool(raw.get("therapeutic_objective")),
            age_min=raw.get("age_min"),
            age_max=raw.get("age_max"),
        )

    def to_clinical_intent(self):
        """Puente hacia rerank / alignment legacy."""
        from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent

        return ClinicalIntent(
            population=list(self.population),
            interventions=list(self.intervention),
            comparator=list(self.comparator),
            outcomes=list(self.outcomes),
            evidence_preference=_preferred_to_legacy(self.preferred_evidence),
            age_min=self.age_min,
            age_max=self.age_max,
            priority_axis=self.priority_axis,  # type: ignore[arg-type]
            question_type=self.question_type,
        )


def _preferred_to_legacy(preferred: list[str]) -> list[str]:
    mapping = {
        "meta_analysis": "meta-analysis",
        "RCT": "rct",
        "guideline": "guideline",
        "systematic_review": "meta-analysis",
    }
    out: list[str] = []
    seen: set[str] = set()
    for p in preferred:
        leg = mapping.get(p, p)
        if leg.lower() not in seen:
            seen.add(leg.lower())
            out.append(leg)
    return out


def _resolve_labels_to_concept_ids(labels: list[str], kind: ConceptKind | None = None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for label in labels:
        cid = resolve_concept_id(label)
        if not cid or cid in seen:
            continue
        if kind is not None:
            c = get_concept(cid)
            if c and c.kind != kind:
                continue
        seen.add(cid)
        ids.append(cid)
    return ids


def build_clinical_evidence_frame(
    query: str,
    clinical_context: Any = None,
) -> ClinicalEvidenceFrame:
    """
    Construye el marco canónico: intent graph + resolución a concept_ids + política única.
    """
    from app.capabilities.evidence_rag.clinical_intent_graph import build_clinical_intent_graph

    graph = build_clinical_intent_graph(query, clinical_context)
    policy = policy_for_question_type(graph.question_type)

    pop_ids = _resolve_labels_to_concept_ids(graph.population, "population")
    iv_ids = _resolve_labels_to_concept_ids(graph.intervention, "intervention")
    comp_ids = _resolve_labels_to_concept_ids(graph.comparator, "comparator")
    out_ids = _resolve_labels_to_concept_ids(graph.outcomes, "outcome")

    landmarks = list(graph.expected_landmark_trials)
    if not landmarks:
        landmarks = expected_landmarks_for_concepts(iv_ids + comp_ids)

    return ClinicalEvidenceFrame(
        question_type=graph.question_type,
        population=list(graph.population),
        intervention=list(graph.intervention),
        comparator=list(graph.comparator),
        outcomes=list(graph.outcomes),
        population_concept_ids=pop_ids,
        intervention_concept_ids=iv_ids,
        comparator_concept_ids=comp_ids,
        outcome_concept_ids=out_ids,
        preferred_evidence=list(policy.preferred_evidence),
        suppress_evidence=list(policy.suppress_evidence),
        expected_landmark_trials=landmarks,
        outcome_theme=graph.outcome_theme,
        priority_axis=graph.priority_axis,
        therapeutic_objective=policy.therapeutic_objective,
        age_min=graph.age_min,
        age_max=graph.age_max,
    )


def frame_from_state(state: Mapping[str, Any]) -> ClinicalEvidenceFrame | None:
    raw = state.get("clinical_evidence_frame")
    if isinstance(raw, ClinicalEvidenceFrame):
        return raw
    if isinstance(raw, dict):
        return ClinicalEvidenceFrame.from_dict(raw)
    raw_g = state.get("clinical_intent_graph")
    if isinstance(raw_g, dict):
        from app.capabilities.evidence_rag.clinical_intent_graph import ClinicalIntentGraph

        g = ClinicalIntentGraph.from_dict(raw_g)
        policy = policy_for_question_type(g.question_type)
        return ClinicalEvidenceFrame(
            question_type=g.question_type,
            population=list(g.population),
            intervention=list(g.intervention),
            comparator=list(g.comparator),
            outcomes=list(g.outcomes),
            preferred_evidence=list(policy.preferred_evidence),
            suppress_evidence=list(policy.suppress_evidence),
            expected_landmark_trials=list(g.expected_landmark_trials),
            outcome_theme=g.outcome_theme,
            priority_axis=g.priority_axis,
            therapeutic_objective=policy.therapeutic_objective,
            age_min=g.age_min,
            age_max=g.age_max,
        )
    return None
