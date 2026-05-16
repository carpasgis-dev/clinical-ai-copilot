"""
Políticas de evidencia por ``question_type`` del Clinical Intent Graph.

Traduce ``suppress_evidence`` / ``preferred_evidence`` a filtros y cláusulas PubMed.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.capabilities.evidence_rag.evidence_rerank import infer_study_type_from_title

if TYPE_CHECKING:
    from app.capabilities.evidence_rag.clinical_intent_graph import ClinicalIntentGraph

_SUPPRESS_PUBMED_TERMS: dict[str, str] = {
    "ehealth": (
        '("telemedicine"[tiab] OR "e-health"[tiab] OR "digital health"[tiab] '
        'OR "ehealth"[tiab])'
    ),
    "prediction_model": (
        '("machine learning"[tiab] OR "deep learning"[tiab] '
        'OR "prediction model"[tiab] OR "predictive model"[tiab])'
    ),
    "epidemiology": (
        '("burden of disease"[tiab] OR "national survey"[tiab] '
        'OR "prevalence study"[tiab])'
    ),
    "preclinical": '("in vitro"[tiab] OR "animal model"[tiab] OR "murine"[tiab])',
    "mechanistic": '("oxidative stress"[tiab] OR "pathophysiology"[tiab] OR "in vitro"[tiab])',
}

_SUPPRESS_TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "ehealth": re.compile(
        r"\b(telemedicine|e-?health|digital health|ehealth integration)\b",
        re.I,
    ),
    "prediction_model": re.compile(
        r"\b(machine learning|deep learning|artificial intelligence|"
        r"prediction model|predictive model|risk score)\b",
        re.I,
    ),
    "epidemiology": re.compile(
        r"\b(burden of disease|national registry description|"
        r"epidemiolog(?:y|ical)\s+survey)\b",
        re.I,
    ),
    "preclinical": re.compile(
        r"\b(in vitro|animal model|murine|zebrafish|preclinical)\b",
        re.I,
    ),
    "mechanistic": re.compile(
        r"\b(oxidative stress|dna damage|pathophysiology only|mechanistic pathway)\b",
        re.I,
    ),
    "editorial": re.compile(r"\b(editorial|letter to the editor|corrigendum)\b", re.I),
    "case_report": re.compile(r"\b(case report|case series)\b", re.I),
    "topical_report": re.compile(
        r"\b(summit report|conference report|workshop report|year in review|"
        r"state of the science|expert panel report)\b",
        re.I,
    ),
}

_STUDY_TYPE_FOR_SUPPRESS: dict[str, frozenset[str]] = {
    "preclinical": frozenset({"basic-research", "preclinical"}),
    "mechanistic": frozenset({"mechanistic-review", "basic-research"}),
    "epidemiology": frozenset({"epidemiology", "cross-sectional"}),
}


def pubmed_noise_exclusion_clause(graph: ClinicalIntentGraph) -> str:
    """Cláusula NOT para queries PubMed según política de supresión."""
    terms: list[str] = []
    for key in graph.suppress_evidence:
        clause = _SUPPRESS_PUBMED_TERMS.get(key)
        if clause and clause not in terms:
            terms.append(clause)
    if not terms:
        return ""
    return f"NOT ({' OR '.join(terms)})"


def article_matches_suppress_policy(
    title: str,
    abstract: str,
    graph: ClinicalIntentGraph,
) -> bool:
    """
    True si el artículo cae en una categoría suprimida por la política del grafo.
    """
    blob = f"{title} {abstract}"
    st = infer_study_type_from_title(title)
    for key in graph.suppress_evidence:
        pats = _STUDY_TYPE_FOR_SUPPRESS.get(key)
        if pats and st in pats:
            return True
        rx = _SUPPRESS_TEXT_PATTERNS.get(key)
        if rx and rx.search(blob):
            if key == "topical_report":
                if re.search(
                    r"\b(randomi[sz]ed|controlled trial|cvot|meta[-\s]?analysis|"
                    r"hazard ratio|primary endpoint)\b",
                    fold_ascii(blob),
                    re.I,
                ):
                    continue
            if key == "mechanistic":
                if re.search(
                    r"\b(randomi[sz]ed|controlled trial|rct|meta[-\s]?analysis|"
                    r"warfarin|doac|apixaban|mace|stroke prevention)\b",
                    fold_ascii(blob),
                    re.I,
                ):
                    continue
            return True
    return False


def passes_graph_evidence_gate(
    title: str,
    abstract: str,
    graph: ClinicalIntentGraph | None,
) -> bool:
    """Filtro pre-rerank basado en política del intent graph."""
    if graph is None:
        return True
    if article_matches_suppress_policy(title, abstract, graph):
        return False
    from app.capabilities.evidence_rag.clinical_answerability import passes_answerability_gate

    intent = graph.to_clinical_intent()
    st = infer_study_type_from_title(title)
    return passes_answerability_gate(
        title,
        abstract,
        study_type=st,
        intent=intent,
        question_type=graph.question_type,
    )
