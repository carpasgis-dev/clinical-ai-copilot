"""Esquema semántico canónico y single source of truth."""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_concepts import (
    DOAC_AGENTS,
    pubmed_phrase_doac,
    pubmed_phrase_sglt2_class,
)
from app.capabilities.evidence_rag.clinical_semantics import (
    CLINICAL_CONCEPTS,
    build_clinical_evidence_frame,
    classify_intervention_in_text,
    policy_for_question_type,
    resolve_concept_id,
)
from app.capabilities.evidence_rag.clinical_intent_graph import build_clinical_intent_graph

_HERO = (
    "En pacientes con diabetes e hipertensión ≥65 años, "
    "¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 "
    "frente a solo metformina para reducir eventos cardiovasculares?"
)


def test_registry_has_core_interventions() -> None:
    assert "sglt2_class" in CLINICAL_CONCEPTS
    assert "glp1_ra" in CLINICAL_CONCEPTS
    assert "doac_class" in CLINICAL_CONCEPTS
    assert len(DOAC_AGENTS) >= 4


def test_resolve_alias_to_canonical_id() -> None:
    assert resolve_concept_id("empagliflozin") == "sglt2_class"
    assert resolve_concept_id("apixaban") == "doac_class"
    assert resolve_concept_id("DOAC") == "doac_class"


def test_policy_single_source_matches_graph() -> None:
    graph = build_clinical_intent_graph(_HERO)
    policy = policy_for_question_type(graph.question_type)
    frame = build_clinical_evidence_frame(_HERO)
    assert frame.question_type == graph.question_type
    assert frame.preferred_evidence == list(policy.preferred_evidence)
    assert frame.suppress_evidence == list(policy.suppress_evidence)
    assert frame.therapeutic_objective is True


def test_frame_resolves_intervention_concept_ids() -> None:
    frame = build_clinical_evidence_frame(_HERO)
    assert "sglt2_class" in frame.intervention_concept_ids or any(
        "sglt2" in x.lower() for x in frame.intervention
    )
    assert frame.therapeutic_objective


def test_pubmed_phrases_from_registry() -> None:
    sglt2 = pubmed_phrase_sglt2_class()
    doac = pubmed_phrase_doac()
    assert "SGLT2" in sglt2 or "gliflozin" in sglt2.lower()
    assert "apixaban" in doac.lower()


def test_classify_intervention_in_abstract() -> None:
    blob = "DECLARE-TIMI 58 dapagliflozin cardiovascular outcomes"
    ids = classify_intervention_in_text(blob)
    assert "sglt2_class" in ids
