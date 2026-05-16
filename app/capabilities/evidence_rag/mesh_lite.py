"""
Fachada de expansión PubMed: concepto clínico → léxico.

Orden de resolución:
1. ``clinical_concepts`` (DOAC, MACE, población ≥65, …)
2. ``lexical_expansion`` (token cohorte / fármaco / enfermedad)

Este módulo NO define ontología ni política de evidencia.
"""
from __future__ import annotations

from app.capabilities.evidence_rag.clinical_concepts import expand_clinical_concept_for_pubmed
from app.capabilities.evidence_rag.lexical_expansion import (
    LEXICAL_TOKEN_TO_PUBMED,
    expand_lexical_token_for_pubmed,
)

# Compatibilidad con imports existentes
TOKEN_TO_EVIDENCE_PHRASE = LEXICAL_TOKEN_TO_PUBMED


def expand_cohort_token_for_pubmed(token: str, is_drug: bool = False) -> str:
    """
    Punto de entrada único para queries: concepto compuesto o expansión léxica.
    """
    concept = expand_clinical_concept_for_pubmed(token)
    if concept:
        return concept
    return expand_lexical_token_for_pubmed(token, is_drug=is_drug)
