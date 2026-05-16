"""
Query de búsqueda de evidencia en texto libre (heurística compartida).

Misma cadena sirve como término para NCBI E-utilities y como ``query`` para Europe PMC REST.
Refactorizado hacia un Retrieval Policy Engine (100% Declarativo y Extensible).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.schemas.copilot_state import ClinicalContext
from app.capabilities.evidence_rag.clinical_intent import (
    ClinicalIntent,
    extract_clinical_intent,
    primary_outcome_theme,
)
from app.capabilities.evidence_rag.mesh_lite import expand_cohort_token_for_pubmed
from app.capabilities.evidence_rag.policies import RETRIEVAL_POLICIES
from app.capabilities.evidence_rag.outcome_ontology import pubmed_clause_for_theme
from app.capabilities.evidence_rag.clinical_knowledge import landmark_pubmed_acronyms_clause, landmark_pubmed_drugs_clause


@dataclass(frozen=True, slots=True)
class RetrievalStage:
    stage_id: str
    query: str


def _norm_question(text: str) -> str:
    from app.capabilities.evidence_rag.intent_semantic_query import build_intent_semantic_query
    return build_intent_semantic_query(None, text)


def build_policy_driven_query(intent: ClinicalIntent, free_text: str, pop_conds: list[str]) -> str:
    """Construcción genérica de query basada puramente en la política correspondiente al theme."""
    theme = primary_outcome_theme(intent)
    policy = RETRIEVAL_POLICIES.get(theme, RETRIEVAL_POLICIES["general"])
    
    blocks = []
    
    # 1. POPULATION
    pop_blocks = []
    for p in intent.population:
        pop_blocks.append(expand_cohort_token_for_pubmed(p, is_drug=False))
    # Fill with structured context if available and intent population is sparse
    if not pop_blocks and pop_conds:
        for c in pop_conds[:2]:
            pop_blocks.append(expand_cohort_token_for_pubmed(c, is_drug=False))
            
    if pop_blocks:
        blocks.append(f"({' OR '.join(pop_blocks)})")
        
    # 2. INTERVENTION
    iv_blocks = []
    for iv in intent.interventions:
        iv_blocks.append(expand_cohort_token_for_pubmed(iv, is_drug=True))
    for iv in intent.comparator:
        if iv.lower() != "metformin":  # Omit comparator to favor recall
            iv_blocks.append(expand_cohort_token_for_pubmed(iv, is_drug=True))
            
    if iv_blocks:
        blocks.append(f"({' OR '.join(iv_blocks)})")
        
    # 3. OUTCOMES (Delegated to policies and ontology)
    if policy.pubmed_broad_clause:
        blocks.append(policy.pubmed_broad_clause)
    else:
        # Fallback to ontology
        clause = pubmed_clause_for_theme(theme, tier="moderate")
        if clause:
            blocks.append(clause)
            
    # Assemble
    core_query = " AND ".join(blocks)
    
    # Inject Landmarks if Policy allows
    if policy.landmark_retrieval_allowed:
        landmark_names = landmark_pubmed_acronyms_clause()
        landmark_drugs = landmark_pubmed_drugs_clause()
        
        candidates = []
        if core_query:
            candidates.append(f"({core_query})")
        if landmark_names and (intent.interventions or intent.comparator):
            candidates.append(f"({landmark_names} AND ({' OR '.join(iv_blocks)}))")
        elif landmark_names:
            candidates.append(f"{landmark_names}")
            
        if candidates:
            return " OR ".join(candidates)
            
    return core_query if core_query else free_text


def build_evidence_search_query(
    free_text: str,
    clinical_context: Optional[Union[ClinicalContext, dict]] = None,
) -> str:
    """Punto de entrada único. Devuelve la query principal."""
    core = " ".join((free_text or "").strip().split())
    if not core:
        return ""

    from app.capabilities.evidence_rag.heuristic_evidence_query import as_clinical_context
    ctx = as_clinical_context(clinical_context)
    if ctx is None:
        return core

    intent = extract_clinical_intent(core, ctx)
    pop_conds = [c for c in (ctx.population_conditions or []) if str(c).strip()]
    
    # Generic builder
    return build_policy_driven_query(intent, core, pop_conds)


def as_clinical_context(obj: Optional[Union[ClinicalContext, dict]]) -> Optional[ClinicalContext]:
    if isinstance(obj, ClinicalContext):
        return obj
    if isinstance(obj, dict):
        return ClinicalContext(**obj)
    return None


def build_evidence_search_queries(
    free_text: str,
    clinical_context: Optional[Union[ClinicalContext, dict]] = None,
) -> list[str]:
    """Retorna listado para el executor (una query monolítica ahora)."""
    q = build_evidence_search_query(free_text, clinical_context)
    return [q] if q else []


def build_evidence_retrieval_stages(
    free_text: str,
    clinical_context: Optional[Union[ClinicalContext, dict]] = None,
) -> list[RetrievalStage]:
    q = build_evidence_search_query(free_text, clinical_context)
    return [RetrievalStage(stage_id="policy_driven_primary", query=q)] if q else []


def pubmed_execute_with_clauses(clauses: Sequence[str]) -> str:
    return " AND ".join(f"({c})" for c in clauses if c.strip())

def preview_pubmed_query(
    free_text: str,
    clinical_context: Optional[Union[ClinicalContext, dict]] = None,
) -> str:
    return build_evidence_search_query(free_text, clinical_context)

def canonical_pubmed_query_after_retrieval(executed_queries: list[str]) -> tuple[str, list[str]]:
    if not executed_queries:
        return "", []
    return executed_queries[0], executed_queries
