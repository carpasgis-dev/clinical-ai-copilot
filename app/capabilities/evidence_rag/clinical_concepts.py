"""
Conceptos clínicos compuestos → cláusulas PubMed.

Lee definiciones del registro canónico ``clinical_semantics``; no duplica listas de fármacos.
"""
from __future__ import annotations

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.capabilities.evidence_rag.clinical_semantics import (
    CLINICAL_CONCEPTS,
    ClinicalConcept,
    get_concept,
    resolve_concept_id,
)
from app.capabilities.evidence_rag.lexical_expansion import expand_lexical_token_for_pubmed

def _members(concept_id: str) -> tuple[str, ...]:
    c = get_concept(concept_id)
    return tuple(c.drug_members) if c else ()


# Compatibilidad hacia atrás (tests / imports legacy)
DOAC_AGENTS = _members("doac_class")
SGLT2_AGENTS = _members("sglt2_class")
GLP1_AGENTS = _members("glp1_ra")


def _or_phrases(parts: list[str]) -> str:
    clean = [p for p in parts if p]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return f"({' OR '.join(clean)})"


def pubmed_phrase_for_concept(concept_id: str) -> str:
    """Cláusula PubMed para un ``ClinicalConcept`` del registro canónico."""
    c = get_concept(concept_id)
    if c is None:
        return ""
    parts: list[str] = []
    for tok in c.lexical_tokens:
        ph = expand_lexical_token_for_pubmed(tok)
        if ph:
            parts.append(ph)
    for drug in c.drug_members:
        ph = expand_lexical_token_for_pubmed(drug, is_drug=True)
        if ph and ph not in parts:
            parts.append(ph)
    if c.pubmed_class_clause:
        parts.append(c.pubmed_class_clause)
    return _or_phrases(parts)


def pubmed_phrase_doac() -> str:
    return pubmed_phrase_for_concept("doac_class")


def pubmed_phrase_sglt2_class() -> str:
    return pubmed_phrase_for_concept("sglt2_class")


def pubmed_phrase_glp1_class() -> str:
    return pubmed_phrase_for_concept("glp1_ra")


def pubmed_phrase_atrial_fibrillation() -> str:
    return pubmed_phrase_for_concept("pop_af")


def pubmed_phrase_older_adults() -> str:
    return pubmed_phrase_for_concept("pop_older_adults")


def pubmed_phrase_major_bleeding() -> str:
    return pubmed_phrase_for_concept("outcome_bleeding")


def pubmed_phrase_systemic_embolism() -> str:
    c = get_concept("outcome_stroke")
    if c:
        return pubmed_phrase_for_concept("outcome_stroke")
    return '("embolism"[Mesh] OR "systemic embolism"[tiab])'


def pubmed_phrase_intracranial_hemorrhage() -> str:
    return '("intracranial hemorrhages"[Mesh] OR "intracranial hemorrhage"[tiab])'


def pubmed_phrase_mace() -> str:
    return pubmed_phrase_for_concept("outcome_mace")


_CONCEPT_BUILDERS: dict[str, callable] = {}


def _register_alias_builders() -> None:
    global _CONCEPT_BUILDERS
    if _CONCEPT_BUILDERS:
        return
    mapping: dict[str, str] = {
        "doac": "doac_class",
        "noac": "doac_class",
        "anticoagulation": "doac_class",
        "direct oral anticoagulant": "doac_class",
        "sglt2": "sglt2_class",
        "glp1": "glp1_ra",
        "glp-1": "glp1_ra",
        "atrial fibrillation": "pop_af",
        "fibrillation": "pop_af",
        "older adults": "pop_older_adults",
        "older adult": "pop_older_adults",
        "elderly": "pop_older_adults",
        "major bleeding": "outcome_bleeding",
        "systemic embolism": "outcome_stroke",
        "mace": "outcome_mace",
        "cardiovascular events": "outcome_mace",
        "vitamin k antagonist": "warfarin",
        "warfarin": "warfarin",
        "metformin": "metformin",
    }
    for alias, cid in mapping.items():
        _CONCEPT_BUILDERS[fold_ascii(alias)] = cid


_register_alias_builders()


def expand_clinical_concept_for_pubmed(token: str) -> str | None:
    key = fold_ascii((token or "").strip())
    if not key:
        return None
    cid = _CONCEPT_BUILDERS.get(key) or resolve_concept_id(token)
    if not cid:
        return None
    phrase = pubmed_phrase_for_concept(cid)
    return phrase if phrase else None
