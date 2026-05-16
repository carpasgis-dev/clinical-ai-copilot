import re
from typing import Any

from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent, intent_asks_cv_outcomes

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
) -> list[dict[str, Any]]:
    """Filtra ruido pre-rerank: dominio CV + ancla diabetes/MACE cuando la pregunta es T2DM-CV."""
    if not articles or not intent:
        return articles

    filtered: list[dict[str, Any]] = []
    cv_intent = intent_asks_cv_outcomes(intent)

    for art in articles:
        txt = f"{art.get('title', '')} {art.get('abstract_snippet', '')}"

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
