"""
Memoria clínica persistente: ensayos landmark, moléculas y priors epistemológicos.

Complementa retrieval heurístico y rerank cuando PubMed no devuelve el trial por
orden de relevancia reciente.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.capabilities.evidence_rag.clinical_intent import ClinicalIntent, intent_asks_cv_outcomes

# ---------------------------------------------------------------------------
# Ontología de CVOTs / trials landmark (T2DM + desenlace CV)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LandmarkTrial:
    acronym: str
    drugs: tuple[str, ...]
    drug_classes: tuple[str, ...]  # sglt2 | glp1
    outcomes: tuple[str, ...]
    population: str
    evidence_level: str
    pubmed_terms: tuple[str, ...]


LANDMARK_CVOTS: tuple[LandmarkTrial, ...] = (
    LandmarkTrial(
        acronym="EMPA-REG OUTCOME",
        drugs=("empagliflozin",),
        drug_classes=("sglt2",),
        outcomes=("cardiovascular death", "mortality", "heart failure hospitalization"),
        population="type 2 diabetes high cardiovascular risk",
        evidence_level="rct_landmark",
        pubmed_terms=("EMPA-REG", "empagliflozin", "BI 10773"),
    ),
    LandmarkTrial(
        acronym="LEADER",
        drugs=("liraglutide",),
        drug_classes=("glp1",),
        outcomes=("cardiovascular death", "mace"),
        population="type 2 diabetes high cardiovascular risk",
        evidence_level="rct_landmark",
        pubmed_terms=("LEADER trial", "LEADER study", "liraglutide"),
    ),
    LandmarkTrial(
        acronym="DECLARE-TIMI 58",
        drugs=("dapagliflozin",),
        drug_classes=("sglt2",),
        outcomes=("mace", "cardiovascular death", "heart failure hospitalization"),
        population="type 2 diabetes",
        evidence_level="rct_landmark",
        pubmed_terms=("DECLARE-TIMI", "DECLARE TIMI", "dapagliflozin"),
    ),
    LandmarkTrial(
        acronym="CANVAS",
        drugs=("canagliflozin",),
        drug_classes=("sglt2",),
        outcomes=("mace", "cardiovascular death"),
        population="type 2 diabetes high cardiovascular risk",
        evidence_level="rct_landmark",
        pubmed_terms=("CANVAS program", "CANVAS trial", "canagliflozin"),
    ),
    LandmarkTrial(
        acronym="REWIND",
        drugs=("dulaglutide",),
        drug_classes=("glp1",),
        outcomes=("mace", "cardiovascular death"),
        population="type 2 diabetes",
        evidence_level="rct_landmark",
        pubmed_terms=("REWIND trial", "REWIND study", "dulaglutide"),
    ),
    LandmarkTrial(
        acronym="SUSTAIN-6",
        drugs=("semaglutide",),
        drug_classes=("glp1",),
        outcomes=("mace", "stroke"),
        population="type 2 diabetes high cardiovascular risk",
        evidence_level="rct_landmark",
        pubmed_terms=("SUSTAIN-6", "SUSTAIN 6", "semaglutide"),
    ),
    LandmarkTrial(
        acronym="SELECT",
        drugs=("semaglutide",),
        drug_classes=("glp1",),
        outcomes=("mace", "cardiovascular death"),
        population="overweight obesity cardiovascular risk",
        evidence_level="rct_landmark",
        pubmed_terms=("SELECT trial", "semaglutide"),
    ),
    LandmarkTrial(
        acronym="CREDENCE",
        drugs=("canagliflozin",),
        drug_classes=("sglt2",),
        outcomes=("renal outcomes", "cardiovascular death"),
        population="type 2 diabetes chronic kidney disease",
        evidence_level="rct_landmark",
        pubmed_terms=("CREDENCE trial", "canagliflozin"),
    ),
    LandmarkTrial(
        acronym="DAPA-HF",
        drugs=("dapagliflozin",),
        drug_classes=("sglt2",),
        outcomes=("heart failure hospitalization", "cardiovascular death"),
        population="heart failure",
        evidence_level="rct_landmark",
        pubmed_terms=("DAPA-HF", "dapagliflozin heart failure"),
    ),
    LandmarkTrial(
        acronym="EMPEROR-Reduced",
        drugs=("empagliflozin",),
        drug_classes=("sglt2",),
        outcomes=("heart failure hospitalization", "cardiovascular death"),
        population="heart failure reduced ejection fraction",
        evidence_level="rct_landmark",
        pubmed_terms=("EMPEROR-Reduced", "EMPEROR Reduced", "empagliflozin"),
    ),
)

_ACRONYM_PATTERNS: list[tuple[re.Pattern[str], LandmarkTrial]] = []
for trial in LANDMARK_CVOTS:
    for term in trial.pubmed_terms:
        pat = re.compile(re.escape(term), re.I)
        _ACRONYM_PATTERNS.append((pat, trial))


def match_landmark_trial(title: str, abstract_snippet: str = "") -> LandmarkTrial | None:
    blob = f"{title} {abstract_snippet}"
    for pat, trial in _ACRONYM_PATTERNS:
        if pat.search(blob):
            return trial
    return None


def landmark_pubmed_acronyms_clause() -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for t in LANDMARK_CVOTS:
        for term in t.pubmed_terms[:2]:
            q = f'"{term}"[tiab]' if " " in term or "-" in term else f"{term}[tiab]"
            if q.lower() not in seen:
                seen.add(q.lower())
                terms.append(q)
    return f"({' OR '.join(terms)})" if terms else ""


# CVOT T2DM con desenlace CV (excluye HF puro tipo DAPA-HF / EMPEROR para no inundar el pool).
_T2DM_CVOT_RETRIEVAL_ACRONYMS: tuple[str, ...] = (
    "EMPA-REG OUTCOME",
    "EMPA-REG",
    "LEADER trial",
    "DECLARE-TIMI 58",
    "DECLARE-TIMI",
    "CANVAS program",
    "REWIND trial",
    "SUSTAIN-6",
    "CREDENCE trial",
)


def landmark_cvot_retrieval_clause() -> str:
    """
    Query PubMed T2: solo nombres de ensayos (sin fármacos sueltos ni AND de clases terapéuticas).
    """
    terms: list[str] = []
    seen: set[str] = set()
    for ac in _T2DM_CVOT_RETRIEVAL_ACRONYMS:
        q = f'"{ac}"[tiab]' if (" " in ac or "-" in ac) else f"{ac}[tiab]"
        if q.lower() not in seen:
            seen.add(q.lower())
            terms.append(q)
    return f"({' OR '.join(terms)})" if terms else ""


def match_diabetes_cvot_landmark(title: str, abstract_snippet: str = "") -> LandmarkTrial | None:
    """Landmark T2DM/CVOT; excluye ensayos HF sin diabetes como ancla principal."""
    trial = match_landmark_trial(title, abstract_snippet)
    if trial is None:
        return None
    if trial.acronym in ("DAPA-HF", "EMPEROR-Reduced"):
        return None
    pop = (trial.population or "").lower()
    if "diabetes" not in pop and trial.acronym in ("SELECT",):
        return None
    return trial


def landmark_pubmed_drugs_clause() -> str:
    drugs: list[str] = []
    seen: set[str] = set()
    for t in LANDMARK_CVOTS:
        for d in t.drugs:
            q = f"{d}[tiab]"
            if q not in seen:
                seen.add(q)
                drugs.append(q)
    return f"({' OR '.join(drugs)})" if drugs else ""


def landmark_rerank_boost(
    title: str,
    abstract_snippet: str = "",
    *,
    clinical_intent: ClinicalIntent | None = None,
) -> float:
    """
    Boost suave 0–0.18 si el artículo coincide con un trial landmark conocido.
    """
    if clinical_intent is not None and not intent_asks_cv_outcomes(clinical_intent):
        return 0.0
    trial = match_diabetes_cvot_landmark(title, abstract_snippet) or match_landmark_trial(
        title, abstract_snippet
    )
    if not trial:
        return 0.0
    if trial.evidence_level == "rct_landmark":
        return 0.28
    return 0.14


def landmark_synthesis_hint(title: str, abstract_snippet: str = "") -> str | None:
    """Texto corto para síntesis cuando se detecta un landmark."""
    trial = match_landmark_trial(title, abstract_snippet)
    if not trial:
        return None
    drugs = ", ".join(trial.drugs)
    outs = ", ".join(trial.outcomes[:3])
    return (
        f"Ensayo landmark conocido ({trial.acronym}): {drugs}; "
        f"población {trial.population}; desenlaces {outs}."
    )


def trials_for_drug_classes(classes: tuple[str, ...]) -> list[LandmarkTrial]:
    want = {c.lower() for c in classes}
    return [t for t in LANDMARK_CVOTS if any(dc in want for dc in t.drug_classes)]
