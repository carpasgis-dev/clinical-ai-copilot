"""Separación lexical_expansion / clinical_concepts / mesh_lite."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_concepts import (
    DOAC_AGENTS,
    expand_clinical_concept_for_pubmed,
    pubmed_phrase_doac,
)
from app.capabilities.evidence_rag.lexical_expansion import (
    LEXICAL_TOKEN_TO_PUBMED,
    expand_lexical_token_for_pubmed,
)
from app.capabilities.evidence_rag.mesh_lite import expand_cohort_token_for_pubmed


def test_lexical_only_has_no_doac_class() -> None:
    assert "doac" not in LEXICAL_TOKEN_TO_PUBMED
    assert "major bleeding" not in LEXICAL_TOKEN_TO_PUBMED


def test_warfarin_lexical_deterministic() -> None:
    a = expand_lexical_token_for_pubmed("warfarin")
    b = expand_lexical_token_for_pubmed("warfarin")
    assert a == b
    assert "Warfarin" in a


def test_doac_is_clinical_concept_not_lexical() -> None:
    phrase = expand_clinical_concept_for_pubmed("DOAC")
    assert phrase
    for agent in DOAC_AGENTS:
        assert agent in phrase.lower()
    assert expand_lexical_token_for_pubmed("doac") == f"doac[tiab]"


def test_mesh_lite_resolves_concept_before_lexical() -> None:
    doac = expand_cohort_token_for_pubmed("doac")
    warfarin = expand_cohort_token_for_pubmed("warfarin")
    assert "apixaban" in doac.lower()
    assert "Warfarin" in warfarin


def test_atrial_fibrillation_via_concept() -> None:
    af = expand_cohort_token_for_pubmed("atrial fibrillation")
    assert "Fibrillation" in af or "fibrillation" in af


def test_doac_phrase_composes_agents() -> None:
    p = pubmed_phrase_doac()
    assert "rivaroxaban" in p
    assert "DOAC" in p or "anticoagulants" in p
