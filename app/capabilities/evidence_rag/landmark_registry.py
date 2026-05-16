"""
Registro de ensayos landmark (CVOT diabetes, anticoagulación FA, etc.).

Responsabilidad única: identidad de trials, términos PubMed y matching en títulos.
Rerank boost y hints de síntesis viven en ``clinical_knowledge`` (capa fina).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.capabilities.clinical_sql.terminology import fold_ascii


@dataclass(frozen=True, slots=True)
class LandmarkTrial:
    acronym: str
    drugs: tuple[str, ...]
    drug_classes: tuple[str, ...]
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

LANDMARK_ANTICOAG_TRIALS: tuple[LandmarkTrial, ...] = (
    LandmarkTrial(
        acronym="ARISTOTLE",
        drugs=("apixaban",),
        drug_classes=("anticoagulation",),
        outcomes=("stroke", "systemic embolism", "major bleeding"),
        population="atrial fibrillation",
        evidence_level="rct_landmark",
        pubmed_terms=("ARISTOTLE trial", "ARISTOTLE study", "apixaban"),
    ),
    LandmarkTrial(
        acronym="RE-LY",
        drugs=("dabigatran",),
        drug_classes=("anticoagulation",),
        outcomes=("stroke", "systemic embolism", "major bleeding"),
        population="atrial fibrillation",
        evidence_level="rct_landmark",
        pubmed_terms=("RE-LY", "RELY trial", "dabigatran"),
    ),
    LandmarkTrial(
        acronym="ROCKET AF",
        drugs=("rivaroxaban",),
        drug_classes=("anticoagulation",),
        outcomes=("stroke", "systemic embolism", "major bleeding"),
        population="atrial fibrillation",
        evidence_level="rct_landmark",
        pubmed_terms=("ROCKET AF", "ROCKET-AF", "rivaroxaban"),
    ),
    LandmarkTrial(
        acronym="ENGAGE AF",
        drugs=("edoxaban",),
        drug_classes=("anticoagulation",),
        outcomes=("stroke", "systemic embolism", "major bleeding"),
        population="atrial fibrillation",
        evidence_level="rct_landmark",
        pubmed_terms=("ENGAGE AF", "ENGAGE-AF", "edoxaban"),
    ),
)

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

_ACRONYM_PATTERNS: list[tuple[re.Pattern[str], LandmarkTrial]] = []


def _register_patterns() -> None:
    global _ACRONYM_PATTERNS
    if _ACRONYM_PATTERNS:
        return
    for trial in (*LANDMARK_CVOTS, *LANDMARK_ANTICOAG_TRIALS):
        for term in trial.pubmed_terms:
            pat = re.compile(re.escape(term), re.I)
            _ACRONYM_PATTERNS.append((pat, trial))


_register_patterns()


def match_landmark_trial(title: str, abstract_snippet: str = "") -> LandmarkTrial | None:
    blob = f"{title} {abstract_snippet}"
    for pat, trial in _ACRONYM_PATTERNS:
        if pat.search(blob):
            return trial
    return None


def match_anticoag_landmark(title: str, abstract_snippet: str = "") -> LandmarkTrial | None:
    trial = match_landmark_trial(title, abstract_snippet)
    if trial and trial.acronym in {t.acronym for t in LANDMARK_ANTICOAG_TRIALS}:
        return trial
    return None


def match_diabetes_cvot_landmark(title: str, abstract_snippet: str = "") -> LandmarkTrial | None:
    trial = match_landmark_trial(title, abstract_snippet)
    if trial is None:
        return None
    if trial.acronym in ("DAPA-HF", "EMPEROR-Reduced"):
        return None
    pop = (trial.population or "").lower()
    if "diabetes" not in pop and trial.acronym in ("SELECT",):
        return None
    return trial


def _terms_to_pubmed_clause(terms: list[str]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        q = f'"{term}"[tiab]' if (" " in term or "-" in term) else f"{term}[tiab]"
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return f"({' OR '.join(out)})" if out else ""


def landmark_pubmed_acronyms_clause() -> str:
    terms: list[str] = []
    for t in LANDMARK_CVOTS:
        terms.extend(t.pubmed_terms[:2])
    return _terms_to_pubmed_clause(terms)


def landmark_cvot_retrieval_clause() -> str:
    return _terms_to_pubmed_clause(list(_T2DM_CVOT_RETRIEVAL_ACRONYMS))


def landmark_anticoag_retrieval_clause() -> str:
    terms: list[str] = []
    for t in LANDMARK_ANTICOAG_TRIALS:
        terms.extend(t.pubmed_terms[:2])
    return _terms_to_pubmed_clause(terms)


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


def trials_for_drug_classes(classes: tuple[str, ...]) -> list[LandmarkTrial]:
    want = {c.lower() for c in classes}
    return [t for t in LANDMARK_CVOTS if any(dc in want for dc in t.drug_classes)]


def expected_acronyms_for_anticoag() -> tuple[str, ...]:
    return tuple(t.acronym for t in LANDMARK_ANTICOAG_TRIALS)
