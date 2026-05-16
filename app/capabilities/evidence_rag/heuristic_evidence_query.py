"""
Query de búsqueda de evidencia en texto libre (heurística compartida).

Misma cadena sirve como término para NCBI E-utilities y como ``query`` para Europe PMC REST.
Refactorizado hacia un Retrieval Policy Engine (100% Declarativo y Extensible).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from app.capabilities.clinical_sql.terminology import fold_ascii
from app.capabilities.evidence_rag.clinical_intent import (
    ClinicalIntent,
    extract_clinical_intent,
    primary_outcome_theme,
)
from app.capabilities.evidence_rag.clinical_intent_graph import (
    ClinicalIntentGraph,
    build_clinical_intent_graph,
)
from app.capabilities.evidence_rag.clinical_knowledge import (
    landmark_anticoag_retrieval_clause,
    landmark_cvot_retrieval_clause,
    landmark_pubmed_acronyms_clause,
)
from app.capabilities.evidence_rag.evidence_policy import pubmed_noise_exclusion_clause
from app.capabilities.evidence_rag.mesh_lite import expand_cohort_token_for_pubmed
from app.capabilities.evidence_rag.outcome_ontology import (
    pubmed_clause_cv_moderate,
    pubmed_clause_cv_primary,
    pubmed_clause_cv_strict,
    pubmed_clause_for_theme,
)
from app.capabilities.evidence_rag.policies import RETRIEVAL_POLICIES
from app.capabilities.evidence_rag.retrieval_tiers import RetrievalTier
from app.schemas.copilot_state import ClinicalContext

_ELDERLY_CLAUSE = '("older adult"[tiab] OR elderly[tiab])'
# Usamos las expansiones robustas (MeSH + tiab) en vez del hardcode tiab-only
_CV_OUTCOME_COMPACT = f"({expand_cohort_token_for_pubmed('cardiac')} OR {expand_cohort_token_for_pubmed('ckd')} OR {expand_cohort_token_for_pubmed('fibrillat')} OR MACE[tiab] OR cardiovascular[tiab])"
_INTERVENTION_COMPACT = f"({expand_cohort_token_for_pubmed('sglt2')} OR {expand_cohort_token_for_pubmed('glp1')})"

@dataclass(frozen=True, slots=True)
class RetrievalStage:
    stage_id: str
    query: str
    run_only_if_pmids_below: int | None = None
    years_back_override: int | None = None
    stop_after_if_pmids_at_least: int | None = None
    stage_role: str = "primary" # primary | supplemental | fallback
    retrieval_tier: int = RetrievalTier.T1_EXACT_PICO


def _broad_pubmed_retrieval_enabled() -> bool:
    v = (os.getenv("COPILOT_BROAD_PUBMED_RETRIEVAL") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _population_blocks(intent: ClinicalIntent, pop_conds: list[str]) -> list[str]:
    pop_blocks: list[str] = []
    for p in intent.population:
        phrase = expand_cohort_token_for_pubmed(p)
        if phrase:
            pop_blocks.append(phrase)
    if not pop_blocks and pop_conds:
        for c in pop_conds[:2]:
            phrase = expand_cohort_token_for_pubmed(c)
            if phrase:
                pop_blocks.append(phrase)
    return pop_blocks


def _intervention_blocks(intent: ClinicalIntent, *, compact: bool = False) -> list[str]:
    blob = " ".join(intent.interventions + intent.comparator).lower()
    if compact and any(
        x in blob
        for x in (
            "sglt2",
            "gliflozin",
            "empagliflozin",
            "dapagliflozin",
            "glp",
            "semaglutide",
            "liraglutide",
            "tirzepatide",
        )
    ):
        return [_INTERVENTION_COMPACT]
    iv_blocks: list[str] = []
    if any(x in blob for x in ("sglt2", "gliflozin", "empagliflozin", "dapagliflozin")):
        iv_blocks.append(_INTERVENTION_COMPACT)
    elif any(x in blob for x in ("glp", "semaglutide", "liraglutide", "tirzepatide")):
        iv_blocks.append(_INTERVENTION_COMPACT)
    for iv in intent.interventions:
        phrase = expand_cohort_token_for_pubmed(iv)
        if phrase and phrase not in iv_blocks:
            iv_blocks.append(phrase)
    for iv in intent.comparator:
        if iv.lower() == "metformin":
            continue
        phrase = expand_cohort_token_for_pubmed(iv)
        if phrase and phrase not in iv_blocks:
            iv_blocks.append(phrase)
    return iv_blocks


def _primary_cv_outcome_clause() -> str:
    """
    Cláusula de desenlace CV para T1.

    ``COPILOT_PRIMARY_CV_CLAUSE``: primary (default) | moderate | strict
    """
    mode = (os.getenv("COPILOT_PRIMARY_CV_CLAUSE") or "primary").strip().lower()
    if mode in ("strict", "narrow"):
        return pubmed_clause_cv_strict()
    if mode in ("moderate", "broad"):
        return pubmed_clause_cv_moderate()
    return pubmed_clause_cv_primary()


def _is_age_population_phrase(phrase: str) -> bool:
    low = phrase.lower()
    return any(x in low for x in ("older adult", "elderly", "aged 65", "olderadult", "geriatric"))


def build_policy_driven_query(intent: ClinicalIntent, free_text: str, pop_conds: list[str]) -> str:
    """Construcción genérica de query basada en la política del theme (puede incluir OR landmark)."""
    theme = primary_outcome_theme(intent)
    policy = RETRIEVAL_POLICIES.get(theme, RETRIEVAL_POLICIES["general"])

    blocks: list[str] = []
    
    # 1. Filtramos explícitamente términos de edad para no restringir artificialmente
    pop_blocks = _population_blocks(intent, pop_conds)
    pop_blocks = [pb for pb in pop_blocks if not _is_age_population_phrase(pb)]
    if pop_blocks:
        limited_pop = pop_blocks[:2]
        for pb in limited_pop:
            blocks.append(pb)

    iv_blocks = _intervention_blocks(intent)
    if iv_blocks:
        blocks.append(f"({' OR '.join(iv_blocks)})")

    if policy.pubmed_broad_clause:
        blocks.append(policy.pubmed_broad_clause)
    else:
        clause = pubmed_clause_for_theme(theme, tier="moderate")
        if clause:
            blocks.append(clause)

    core_query = " AND ".join(blocks)

    if policy.landmark_retrieval_allowed:
        landmark_names = landmark_pubmed_acronyms_clause()
        candidates: list[str] = []
        if core_query:
            candidates.append(f"({core_query})")
        if landmark_names and iv_blocks:
            candidates.append(f"({landmark_names} AND ({' OR '.join(iv_blocks)}))")
        elif landmark_names:
            candidates.append(landmark_names)
        if candidates:
            return " OR ".join(candidates)

    return core_query if core_query else free_text


def _primary_population_clause(
    intent: ClinicalIntent,
    pop_conds: list[str],
    free_text: str = "",
) -> str | None:
    """
    Un solo bloque de población en T1 para no hundir recall con AND(diabetes, HTA, edad…).

    La hipertensión y la edad se usan en rerank/aplicabilidad, no como AND bloqueante en PubMed.
    Si la pregunta enfatiza otra condición (p. ej. FA), priorizar esa frente a diabetes de cohorte.
    """
    focus = fold_ascii(free_text or "")
    best_score = -1
    best_phrase: str | None = None
    for label in list(intent.population) + pop_conds:
        low = (label or "").strip().lower()
        if not low or _is_age_population_phrase(label):
            continue
        if "hipertens" in low or "hypertension" in low or "hta" in low:
            continue
        phrase = expand_cohort_token_for_pubmed(label)
        if not phrase:
            continue
        score = 0
        if "fibril" in low or "atrial" in low or "auricular" in low:
            if any(x in focus for x in ("fibril", "auricular", "atrial", "ictus", "stroke")):
                score += 10
        if "diabet" in low and "diabet" in focus:
            score += 8
        elif "diabet" in phrase.lower():
            score += 3
        if score > best_score:
            best_score = score
            best_phrase = phrase
    if best_phrase and best_score > 0:
        return best_phrase
    for label in list(intent.population) + pop_conds:
        low = (label or "").strip().lower()
        if not low or _is_age_population_phrase(label):
            continue
        if "hipertens" in low or "hypertension" in low or "hta" in low:
            continue
        phrase = expand_cohort_token_for_pubmed(label)
        if phrase and "diabet" in phrase.lower():
            return phrase
    pop_blocks = [
        pb
        for pb in _population_blocks(intent, pop_conds)
        if not _is_age_population_phrase(pb)
        and "hypertens" not in pb.lower()
    ]
    return pop_blocks[0] if pop_blocks else None


def build_broad_primary_query(
    intent: ClinicalIntent,
    free_text: str,
    pop_conds: list[str],
    ctx: ClinicalContext | None,
) -> str:
    """Query T1: diabetes + intervención + desenlace CV acotado (evitar caer en fallback tier 4)."""
    theme = primary_outcome_theme(intent)
    policy = RETRIEVAL_POLICIES.get(theme, RETRIEVAL_POLICIES["general"])

    blocks: list[str] = []

    pop_clause = _primary_population_clause(intent, pop_conds, free_text)
    if pop_clause:
        blocks.append(pop_clause)

    iv_blocks = _intervention_blocks(intent, compact=True)
    if iv_blocks:
        blocks.append(iv_blocks[0] if len(iv_blocks) == 1 else f"({' OR '.join(iv_blocks)})")

    if theme == "cv":
        blocks.append(_primary_cv_outcome_clause())
    elif policy.pubmed_broad_clause:
        blocks.append(policy.pubmed_broad_clause)
    else:
        clause = pubmed_clause_for_theme(theme, tier="moderate")
        if clause:
            blocks.append(clause)

    return " AND ".join(blocks) if blocks else free_text


def build_cvot_landmark_query(intent: ClinicalIntent) -> str:
    """Etapa T2: solo acrónimos de ensayos T2DM-CVOT (sin empagliflozin[tiab] ni AND SGLT2/GLP-1)."""
    return landmark_cvot_retrieval_clause()


def _graph_iv_pubmed_blocks(graph: ClinicalIntentGraph) -> list[str]:
    blocks: list[str] = []
    for label in graph.intervention:
        phrase = expand_cohort_token_for_pubmed(label)
        if phrase and phrase not in blocks:
            blocks.append(phrase)
    return blocks


def _graph_outcome_pubmed_block(graph: ClinicalIntentGraph) -> str:
    parts: list[str] = []
    for o in graph.outcomes:
        phrase = expand_cohort_token_for_pubmed(o)
        if phrase:
            parts.append(phrase)
    if not parts:
        return (
            '("stroke"[tiab] OR "systemic embolism"[tiab] OR "major bleeding"[tiab] '
            'OR "intracranial hemorrhage"[tiab])'
        )
    return f"({' OR '.join(parts)})"


def build_comparative_anticoag_primary_query(graph: ClinicalIntentGraph) -> str:
    """T1 comparativa DOAC vs warfarina: población FA + intervención + desenlace + diseño."""
    blocks: list[str] = []
    if graph.population:
        pop_phrases = [
            expand_cohort_token_for_pubmed(p)
            for p in graph.population[:2]
            if expand_cohort_token_for_pubmed(p)
        ]
        if pop_phrases:
            blocks.append(f"({' OR '.join(pop_phrases)})")

    iv_blocks = _graph_iv_pubmed_blocks(graph)
    if iv_blocks:
        blocks.append(f"({' OR '.join(iv_blocks)})")

    comp_blocks = [
        expand_cohort_token_for_pubmed(c)
        for c in graph.comparator
        if expand_cohort_token_for_pubmed(c)
    ]
    if comp_blocks:
        blocks.append(f"({' OR '.join(comp_blocks)})")

    blocks.append(_graph_outcome_pubmed_block(graph))
    blocks.append(
        '("Randomized Controlled Trial"[pt] OR "Meta-Analysis"[pt] OR '
        '"Systematic Review"[pt] OR "Practice Guideline"[pt])'
    )
    core = " AND ".join(blocks)
    noise = pubmed_noise_exclusion_clause(graph)
    return f"{core} {noise}".strip() if noise else core


def build_landmark_rescue_query_for_graph(graph: ClinicalIntentGraph) -> str:
    """Query de rescate cuando faltan landmarks esperados en el pool."""
    from app.capabilities.evidence_rag.clinical_intent_graph import _is_anticoag_comparative

    if _is_anticoag_comparative(graph):
        return build_anticoag_landmark_rescue_query(graph)
    if graph.expected_landmark_trials and graph.outcome_theme == "cv":
        return landmark_cvot_retrieval_clause()
    return ""


def build_anticoag_landmark_rescue_query(graph: ClinicalIntentGraph) -> str:
    """Rescate: ensayos landmark de anticoagulación cuando faltan en el pool."""
    land = landmark_anticoag_retrieval_clause()
    if not land:
        return ""
    pop = expand_cohort_token_for_pubmed("atrial fibrillation")
    if pop:
        return f"({land}) AND ({pop})"
    return land


def build_evidence_retrieval_stages_for_graph(
    graph: ClinicalIntentGraph,
    free_text: str,
    pop_conds: list[str],
    ctx: ClinicalContext | None,
) -> list[RetrievalStage]:
    """Plan de recuperación guiado por Clinical Intent Graph."""
    intent = graph.to_clinical_intent()
    theme = graph.outcome_theme or primary_outcome_theme(intent)

    if graph.question_type in ("comparative_effectiveness", "treatment_selection"):
        from app.capabilities.evidence_rag.clinical_intent_graph import _is_anticoag_comparative

        if _is_anticoag_comparative(graph):
            stages: list[RetrievalStage] = []
            primary = build_comparative_anticoag_primary_query(graph)
            if primary:
                stages.append(
                    RetrievalStage(
                        stage_id="anticoag_comparative_primary",
                        query=primary,
                        stop_after_if_pmids_at_least=15,
                        stage_role="primary",
                        retrieval_tier=RetrievalTier.T1_EXACT_PICO,
                    )
                )
            land_q = build_anticoag_landmark_rescue_query(graph)
            if land_q:
                stages.append(
                    RetrievalStage(
                        stage_id="anticoag_landmark",
                        query=land_q,
                        years_back_override=20,
                        stage_role="supplemental",
                        retrieval_tier=RetrievalTier.T2_LANDMARK_CVOT,
                    )
                )
            if stages:
                return stages

    if theme == "safety" or graph.question_type == "adverse_effect":
        q = build_policy_driven_query(intent, free_text, pop_conds)
        return (
            [RetrievalStage(stage_id="safety_primary", query=q, retrieval_tier=RetrievalTier.T1_EXACT_PICO)]
            if q
            else []
        )

    if graph.question_type == "guideline_lookup":
        g_blocks = _graph_iv_pubmed_blocks(graph) or _population_blocks(intent, pop_conds)
        clause = (
            '("Practice Guideline"[pt] OR "Guideline"[pt] OR "consensus"[tiab] OR '
            '"position statement"[tiab])'
        )
        if g_blocks:
            q = f"({' OR '.join(g_blocks[:2])}) AND {clause}"
            noise = pubmed_noise_exclusion_clause(graph)
            if noise:
                q = f"{q} {noise}"
            return [RetrievalStage(stage_id="guideline_primary", query=q, retrieval_tier=RetrievalTier.T1_EXACT_PICO)]

    return []


def build_evidence_search_query(
    free_text: str,
    clinical_context: Optional[Union[ClinicalContext, dict]] = None,
) -> str:
    """Punto de entrada único. Devuelve la query principal (primera etapa de recuperación)."""
    stages = build_evidence_retrieval_stages(free_text, clinical_context)
    if stages:
        return stages[0].query
    core = " ".join((free_text or "").strip().split())
    return core


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
    """Queries ejecutables (una por etapa planificada)."""
    return [s.query for s in build_evidence_retrieval_stages(free_text, clinical_context) if s.query]


def build_evidence_retrieval_stages(
    free_text: str,
    clinical_context: Optional[Union[ClinicalContext, dict]] = None,
) -> list[RetrievalStage]:
    core = " ".join((free_text or "").strip().split())
    if not core:
        return []

    ctx = as_clinical_context(clinical_context)
    graph = build_clinical_intent_graph(core, ctx)
    intent = graph.to_clinical_intent()
    pop_conds = [c for c in (ctx.population_conditions or []) if str(c).strip()] if ctx else []
    theme = graph.outcome_theme or primary_outcome_theme(intent)

    graph_stages = build_evidence_retrieval_stages_for_graph(graph, core, pop_conds, ctx)
    if graph_stages:
        return graph_stages

    if not _broad_pubmed_retrieval_enabled():
        q = build_policy_driven_query(intent, core, pop_conds)
        return [RetrievalStage(stage_id="policy_driven_primary", query=q, retrieval_tier=RetrievalTier.T1_EXACT_PICO)] if q else []

    if theme == "safety":
        q = build_policy_driven_query(intent, core, pop_conds)
        return [RetrievalStage(stage_id="safety_primary", query=q, retrieval_tier=RetrievalTier.T1_EXACT_PICO)] if q else []

    broad_q = build_broad_primary_query(intent, core, pop_conds, ctx)
    stages: list[RetrievalStage] = []
    if broad_q:
        stages.append(
            RetrievalStage(
                stage_id="broad_primary",
                query=broad_q,
                stop_after_if_pmids_at_least=20,
                stage_role="primary",
                retrieval_tier=RetrievalTier.T1_EXACT_PICO,
            )
        )

    if theme == "cv":
        land_q = build_cvot_landmark_query(intent)
        if land_q:
            stages.append(
                RetrievalStage(
                    stage_id="cvot_landmark",
                    query=land_q,
                    years_back_override=15,
                    stage_role="supplemental",
                    retrieval_tier=RetrievalTier.T2_LANDMARK_CVOT,
                )
            )

        iv_blocks = _intervention_blocks(intent, compact=True)
        if iv_blocks:
            sys_blocks = [
                iv_blocks[0] if len(iv_blocks) == 1 else f"({' OR '.join(iv_blocks)})",
                pubmed_clause_cv_moderate(),
                '("Meta-Analysis"[pt] OR "Systematic Review"[pt] OR "Randomized Controlled Trial"[pt])',
            ]
            stages.append(
                RetrievalStage(
                    stage_id="systematic_cv",
                    query=" AND ".join(sys_blocks),
                    run_only_if_pmids_below=12,
                    stage_role="supplemental",
                    retrieval_tier=RetrievalTier.T3_SYSTEMATIC,
                )
            )

        # Fallback tier 4: último recurso (no tocar lógica; sólo si el pool sigue vacío)
        relaxed_blocks: list[str] = []
        if iv_blocks:
            relaxed_blocks.append(
                f"({' OR '.join(iv_blocks)})" if len(iv_blocks) > 1 else iv_blocks[0]
            )
        relaxed_blocks.append(_CV_OUTCOME_COMPACT)
        if len(relaxed_blocks) >= 2:
            stages.append(
                RetrievalStage(
                    stage_id="broad_relaxed",
                    query=" AND ".join(relaxed_blocks),
                    run_only_if_pmids_below=5,
                    stage_role="fallback",
                    retrieval_tier=RetrievalTier.T4_SEMANTIC_BROAD,
                )
            )

    if not stages:
        q = build_policy_driven_query(intent, core, pop_conds)
        if q:
            stages.append(RetrievalStage(stage_id="policy_driven_primary", query=q, stage_role="primary", retrieval_tier=RetrievalTier.T1_EXACT_PICO))
    return stages


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
