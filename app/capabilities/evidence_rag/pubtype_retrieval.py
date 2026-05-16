"""
Recuperación con prior epistemológico (Publication Type) como boost suave.

No filtra exclusión: etapa suplementaria que añade ECA, meta-análisis y guías al pool.
"""
from __future__ import annotations

from app.capabilities.evidence_rag.outcome_ontology import pubmed_clause_cv_moderate

# Tipos de publicación de alta jerarquía (OR — no AND obligatorio en toda la query).
PUBTYPE_HIGH_EVIDENCE_OR = (
    '("Randomized Controlled Trial"[Publication Type] OR '
    '"Meta-Analysis"[Publication Type] OR '
    '"Systematic Review"[Publication Type] OR '
    '"Practice Guideline"[Publication Type] OR '
    '"Guideline"[Publication Type])'
)


def pubmed_evidence_hierarchy_clause(*, include_outcome: bool = True) -> str:
    """
    Bloque OR para stage suplementario: diseño fuerte O desenlace CV moderado.

    Se combina con población+intervención en AND (recall de RCT/guías con señal CV).
    """
    parts = [PUBTYPE_HIGH_EVIDENCE_OR]
    if include_outcome:
        oc = pubmed_clause_cv_moderate()
        if oc:
            parts.append(oc)
    return f"({' OR '.join(parts)})"
