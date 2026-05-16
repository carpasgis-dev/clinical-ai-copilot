import re
from typing import Any

from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent, intent_asks_cv_outcomes
from app.capabilities.evidence_rag.clinical_intent_graph import ClinicalIntentGraph
from app.capabilities.evidence_rag.evidence_policy import passes_graph_evidence_gate
from app.capabilities.evidence_rag.evidence_rerank import infer_study_type_from_title

_CV_KEYWORDS = re.compile(
    r"\b(mace|myocardial infarction|stroke|cardiovascular death|heart failure|heart disease|"
    r"kidney disease|renal failure|ckd|atrial fibrillation|cardiac|atherosclerotic|"
    r"cardiovascular outcomes?|cvot|major adverse cardiovascular)\b",
    re.IGNORECASE,
)

_DIABETES_ANCHOR = re.compile(
    r"\b(type\s*2\s*diabet|t2dm|diabetes\s*mellitus|diabetic\s+patients?)\b",
    re.IGNORECASE,
)

_MACE_ANCHOR = re.compile(
    r"\b(mace|major adverse cardiovascular|cardiovascular death|myocardial infarction|"
    r"stroke|cvot|cardiovascular outcome)\b",
    re.IGNORECASE,
)

_MECHANISTIC_ONLY = re.compile(
    r"\b(fibrosis|mechanistic|pathophysiology|steatosis|nafld|nash|oxidative stress)\b",
    re.IGNORECASE,
)

_HF_PHENOTYPE_ONLY = re.compile(
    r"\b(hfpef|hfref|heart failure with preserved|heart failure with reduced|"
    r"ejection fraction)\b",
    re.IGNORECASE,
)

_AKI_ONLY = re.compile(
    r"\bacute kidney injury\b|\baki\b",
    re.IGNORECASE,
)


def filter_off_topic_abstracts(
    articles: list[dict[str, Any]],
    intent: ClinicalIntent | None,
    *,
    intent_graph: ClinicalIntentGraph | None = None,
) -> list[dict[str, Any]]:
    """Filtra ruido pre-rerank según política del intent graph y dominio CV."""
    if not articles or (not intent and not intent_graph):
        return articles

    graph = intent_graph
    if graph is None and intent is not None:
        from app.capabilities.evidence_rag.clinical_intent_graph import ClinicalIntentGraph

        graph = ClinicalIntentGraph(
            question_type=intent.question_type,
            population=list(intent.population),
            intervention=list(intent.interventions),
            comparator=list(intent.comparator),
            outcomes=list(intent.outcomes),
            suppress_evidence=[],
            preferred_evidence=list(intent.evidence_preference),
            priority_axis=intent.priority_axis,
            age_min=intent.age_min,
            age_max=intent.age_max,
            population_noise=list(intent.population_noise),
        )

    filtered: list[dict[str, Any]] = []
    cv_intent = intent is not None and intent_asks_cv_outcomes(intent)

    for art in articles:
        txt = f"{art.get('title', '')} {art.get('abstract_snippet', '')}"
        tit = str(art.get("title") or "")
        snip = str(art.get("abstract_snippet") or "")

        if not passes_graph_evidence_gate(tit, snip, graph):
            continue

        if cv_intent:
            if not _CV_KEYWORDS.search(txt):
                continue
            if _MECHANISTIC_ONLY.search(txt) and not _MACE_ANCHOR.search(txt):
                continue
            if _HF_PHENOTYPE_ONLY.search(txt) and not _DIABETES_ANCHOR.search(txt):
                continue
            if _AKI_ONLY.search(txt) and not _MACE_ANCHOR.search(txt):
                continue
            if not _DIABETES_ANCHOR.search(txt) and not _MACE_ANCHOR.search(txt):
                continue

        filtered.append(art)

    return filtered if filtered else articles
