
"""
Ejecutor del plan (fase 3.6): un solo nodo LangGraph recorre ``ExecutionPlan`` y aplica
los mismos efectos que antes repartían varios nodos (trazas y reducers intactos).
"""
from __future__ import annotations

from app.config.settings import settings

from typing import Any

from app.capabilities.contracts import ClinicalCapability, EvidenceCapability
from app.capabilities.evidence_rag.copilot_errors import CopilotError
from app.capabilities.evidence_rag.clinical_intent_graph import (
    build_clinical_intent_graph,
    landmarks_found_in_articles,
)
from app.capabilities.evidence_rag.clinical_semantics import build_clinical_evidence_frame
from app.capabilities.evidence_rag.evidence_pool_filter import filter_off_topic_abstracts
from app.capabilities.evidence_rag.evidence_rerank import (
    clinical_weak_evidence_share,
    rerank_article_dicts,
)
from app.capabilities.evidence_rag.heuristic_evidence_query import (
    RetrievalStage,
    as_clinical_context,
    build_evidence_retrieval_stages,
    build_evidence_search_queries,
    build_landmark_rescue_query_for_graph,
    canonical_pubmed_query_after_retrieval,
    preview_pubmed_query,
)
from app.capabilities.evidence_rag.query_planning import pubmed_llm_refine_enabled
from app.capabilities.evidence_rag.query_planning.llm_planner import LlmQueryPlanner
from app.capabilities.evidence_rag.retrieval_parallel import (
    gather_sync_calls_blocking,
    parallel_retrieval_enabled,
    partial_call,
)
from app.orchestration.planner import PlanStep, build_execution_plan
from app.orchestration.nodes import (
    _graph_effective_user_query,
    _new_trace_step,
    clarify_node,
    hybrid_clinical_route_node,
    hybrid_pubmed_route_node,
    sql_route_node,
    unknown_stub_node,
)
from app.schemas.copilot_state import (
    CopilotState,
    EvidenceBundle,
    NodeName,
    Route,

)


def _apply_delta(
    work: dict[str, Any],
    delta: dict[str, Any],
    acc_trace: list,
    acc_warn: list,
) -> None:
    acc_trace.extend(delta.get("trace") or [])
    acc_warn.extend(delta.get("warnings") or [])
    for k, v in delta.items():
        if k not in ("trace", "warnings"):
            work[k] = v


def _step_pubmed_evidence_route(
    work: dict[str, Any], evidence: EvidenceCapability
) -> dict[str, Any]:
    query = _graph_effective_user_query(work)
    built = preview_pubmed_query(query, work.get("clinical_context"))
    warn: list[str] = []
    if not (built or "").strip():
        warn.append("evidence: vista previa PubMed heurística vacía")
    return {
        "pubmed_query": built,
        "trace": _new_trace_step(
            NodeName.PUBMED_QUERY_BUILDER,
            "heuristic preview (canonical query set after evidence_retrieval)",
        ),
        "warnings": warn,
    }


def _merge_bundle_articles(
    bundle: EvidenceBundle,
    stage_id: str,
    retrieval_tier: int,
    *,
    merged_arts: list[dict[str, Any]],
    seen_pm: dict[str, dict[str, Any]],
) -> None:
    for art in bundle.articles:
        d = art.model_dump(mode="json") if hasattr(art, "model_dump") else dict(art)
        pm = str(d.get("pmid") or "").strip()
        if not pm:
            continue
            
        if pm in seen_pm:
            # Si ya existía, vemos si la nueva etapa tiene un mejor tier (menor número)
            existing_art = seen_pm[pm]
            current_tier = existing_art.get("retrieval_tier", 99)
            if retrieval_tier < current_tier:
                existing_art["retrieval_tier"] = retrieval_tier
                existing_art["retrieval_stage"] = stage_id
            continue
            
        d["retrieval_stage"] = stage_id
        d["retrieval_tier"] = retrieval_tier
        seen_pm[pm] = d
        merged_arts.append(d)


def _run_retrieval_stages_parallel(
    stage_plan: list[RetrievalStage],
    *,
    evidence: EvidenceCapability,
    years_back: int,
    pool_max: int,
    merged_arts: list[dict[str, Any]],
    seen_pm: dict[str, dict[str, Any]],
    queries: list[str],
    stages_executed: list[dict[str, str]],
    primary_bundle_holder: list[EvidenceBundle | None],
) -> None:
    """Ejecuta todas las etapas con query en paralelo (``asyncio.gather``)."""
    calls: list = []
    stages_for_call: list[RetrievalStage] = []
    retmax_each = max(10, pool_max)

    for stage in stage_plan:
        q = (stage.query or "").strip()
        if not q:
            continue
        eff_years = (
            stage.years_back_override
            if stage.years_back_override is not None
            else years_back
        )
        calls.append(
            partial_call(
                evidence.retrieve_evidence,
                q,
                retmax=retmax_each,
                years_back=eff_years,
            )
        )
        stages_for_call.append(stage)

    if not calls:
        return

    bundles = gather_sync_calls_blocking(calls)
    for stage, bundle in zip(stages_for_call, bundles, strict=True):
        q = (stage.query or "").strip()
        if primary_bundle_holder[0] is None:
            primary_bundle_holder[0] = bundle
            
        rd = bundle.retrieval_debug or {}
        stages_executed.append({
            "stage_id": stage.stage_id, 
            "query": q,
            "pmids": bundle.pmids,
            "articles_count": len(bundle.articles),
            "outcome": rd.get("outcome", "unknown"),
            "raw_debug": rd
        })
        
        if q not in queries:
            queries.append(q)
        
        # Leemos el tier si está, sino fallback a 1
        tier = getattr(stage, "retrieval_tier", 1)
        _merge_bundle_articles(bundle, stage.stage_id, tier, merged_arts=merged_arts, seen_pm=seen_pm)
        if len(merged_arts) >= pool_max:
            break


def _run_retrieval_stage(
    stage: RetrievalStage,
    *,
    evidence: EvidenceCapability,
    years_back: int,
    pool_max: int,
    merged_arts: list[dict[str, Any]],
    seen_pm: dict[str, dict[str, Any]],
    queries: list[str],
    stages_executed: list[dict[str, str]],
    primary_bundle_holder: list[EvidenceBundle | None],
) -> None:
    q = (stage.query or "").strip()
    if not q:
        return
    if stage.run_only_if_pmids_below is not None and len(merged_arts) >= stage.run_only_if_pmids_below:
        return
    if len(merged_arts) >= pool_max:
        return
    retmax_this = max(10, min(pool_max, pool_max - len(merged_arts)))
    eff_years = (
        stage.years_back_override if stage.years_back_override is not None else years_back
    )
    refine_pubtypes = stage.stage_id not in ("cvot_landmark", "broad_relaxed")
    bundle = evidence.retrieve_evidence(
        q,
        retmax=retmax_this,
        years_back=eff_years,
        synthesis_pubtype_refine=refine_pubtypes,
    )
    if primary_bundle_holder[0] is None:
        primary_bundle_holder[0] = bundle
        
    rd = bundle.retrieval_debug or {}
    stages_executed.append({
        "stage_id": stage.stage_id, 
        "query": q,
        "pmids": bundle.pmids,
        "articles_count": len(bundle.articles),
        "outcome": rd.get("outcome", "unknown"),
        "raw_debug": rd
    })
    
    if q not in queries:
        queries.append(q)
    
    tier = getattr(stage, "retrieval_tier", 1)
    _merge_bundle_articles(bundle, stage.stage_id, tier, merged_arts=merged_arts, seen_pm=seen_pm)


def _maybe_llm_refine_stage(
    *,
    evidence: EvidenceCapability,
    uq: str,
    ctx_raw: Any,
    years_back: int,
    pool_max: int,
    merged_arts: list[dict[str, Any]],
    seen_pm: set[str],
    queries: list[str],
    stages_executed: list[dict[str, str]],
    primary_bundle_holder: list[EvidenceBundle | None],
    warn: list[str],
) -> None:
    if not pubmed_llm_refine_enabled():
        return
    if len(merged_arts) >= pool_max:
        return
    try:
        llm_q = LlmQueryPlanner().build_query(uq, ctx_raw).strip()
    except CopilotError as exc:
        warn.append(f"evidence: refinamiento LLM omitido ({exc.code})")
        return
    if not llm_q or llm_q in queries:
        return
    remaining = pool_max - len(merged_arts)
    bundle = evidence.retrieve_evidence(
        llm_q,
        retmax=max(10, min(40, remaining)),
        years_back=years_back,
    )
    if primary_bundle_holder[0] is None:
        primary_bundle_holder[0] = bundle
        
    rd = bundle.retrieval_debug or {}
    stages_executed.append({
        "stage_id": "llm_refine", 
        "query": llm_q,
        "pmids": bundle.pmids,
        "articles_count": len(bundle.articles),
        "outcome": rd.get("outcome", "unknown"),
        "raw_debug": rd
    })
    queries.append(llm_q)
    _merge_bundle_articles(bundle, merged_arts=merged_arts, seen_pm=seen_pm)
    warn.append(
        "evidence: etapa opcional llm_refine (COPILOT_PUBMED_LLM_REFINE) añadió candidatos."
    )


# Híbrido: ventana PDAT algo más amplia que evidencia pura (preguntas clínicas suelen tolerar >5 años).
_HYBRID_PUBMED_YEARS_BACK = 12  # cubre EMPA-REG (sep 2015) y LEADER (jun 2016)
_EVIDENCE_PUBMED_YEARS_BACK = 5
# Candidatos pre-rerank (por query); el rerank devuelve 6. Debe superar _EVIDENCE_MAX_ART.
_EVIDENCE_RETRIEVAL_CANDIDATES = settings.evidence_retrieval_pool_max


def _cohort_rerank_hints(
    work: dict[str, Any],
) -> tuple[int | None, list[str] | None, list[str] | None]:
    """Edad mínima, condiciones y medicación de cohorte estructurada para el re-ranking."""
    cc = work.get("clinical_context")
    if isinstance(cc, dict):
        pop_age_min: int | None = None
        try:
            v = cc.get("population_age_min")
            if v is not None:
                pop_age_min = int(v)
        except (TypeError, ValueError):
            pop_age_min = None
        pc = cc.get("population_conditions")
        conds: list[str] | None = None
        if isinstance(pc, list):
            conds = [str(x).strip() for x in pc if str(x).strip()] or None
        pm = cc.get("population_medications")
        meds: list[str] | None = None
        if isinstance(pm, list):
            meds = [str(x).strip() for x in pm if str(x).strip()] or None
        return pop_age_min, conds, meds
    if cc is not None and hasattr(cc, "population_age_min"):
        pop_age_min = None
        try:
            v = getattr(cc, "population_age_min", None)
            if v is not None:
                pop_age_min = int(v)
        except (TypeError, ValueError):
            pop_age_min = None
        pc = getattr(cc, "population_conditions", None) or []
        conds = [str(x).strip() for x in pc if str(x).strip()] or None
        pm = getattr(cc, "population_medications", None) or []
        meds = [str(x).strip() for x in pm if str(x).strip()] or None
        return pop_age_min, conds, meds
    return None, None, None


def _aggregate_retrieval_debug(
    original_rd: dict[str, Any] | None,
    stages_executed: list[dict[str, Any]],
    merged_arts: list[dict[str, Any]]
) -> dict[str, Any]:
    out = dict(original_rd) if original_rd else {}
    
    total_pmids = len(merged_arts)
    if total_pmids == 0:
        outcome = "zero_hits_all_stages"
    else:
        # Check if the primary stage got PMIDs
        if stages_executed and not stages_executed[0].get("pmids"):
            outcome = "partial_primary_miss"
        else:
            outcome = "success"
            
    out["outcome"] = outcome
    
    # Merge attempts from all stages
    all_attempts = []
    all_errors = []
    
    for s_exec in stages_executed:
        raw_db = s_exec.get("raw_debug") or {}
        st_attempts = raw_db.get("attempts", [])
        st_errors = raw_db.get("errors", [])
        
        # Inject stage_id and pmids fetched into attempts
        for att in st_attempts:
            att_copy = dict(att)
            att_copy["stage_id"] = s_exec.get("stage_id")
            att_copy["pmids_from_stage"] = s_exec.get("pmids", [])
            att_copy["articles_from_stage"] = s_exec.get("articles_count", 0)
            all_attempts.append(att_copy)
            
        all_errors.extend(st_errors)
        
    out["attempts"] = all_attempts
    # Remove duplicates from errors
    out["errors"] = list(dict.fromkeys(all_errors))
    
    return out


def _step_evidence_retrieve(
    work: dict[str, Any],
    evidence: EvidenceCapability,
    *,
    trace_summary: str,
    years_back: int = _EVIDENCE_PUBMED_YEARS_BACK,
) -> dict[str, Any]:
    uq = _graph_effective_user_query(work)
    planned = (work.get("pubmed_query") or uq or "").strip()
    warn: list[str] = []

    ctx_raw = work.get("clinical_context")
    intent_graph = build_clinical_intent_graph(uq, ctx_raw)
    evidence_frame = build_clinical_evidence_frame(uq, ctx_raw)
    retrieval_stages: list[RetrievalStage] = build_evidence_retrieval_stages(uq, ctx_raw)
    adaptive = bool(
        retrieval_stages
        and retrieval_stages[0].stage_id not in ("legacy",)
    )

    if adaptive:
        stage_plan = retrieval_stages
    else:
        stage_queries = build_evidence_search_queries(uq, ctx_raw)
        queries_flat: list[str] = []
        for q in stage_queries:
            if q and q not in queries_flat:
                queries_flat.append(q)
        if not queries_flat:
            queries_flat = [planned] if planned else []
        stage_plan = [RetrievalStage(stage_id=f"flat_{i}", query=q) for i, q in enumerate(queries_flat)]

    queries: list[str] = []
    stages_executed: list[dict[str, str]] = []
    pool_max = settings.evidence_retrieval_pool_max
    merged_arts: list[dict[str, Any]] = []
    seen_pm: dict[str, dict[str, Any]] = {}
    primary_holder: list[EvidenceBundle | None] = [None]
    
    # Desactivamos paralelo si el plan es en escalera (ladder) e incluye fallbacks o múltiples roles
    has_ladder_logic = any(getattr(s, "stage_role", "primary") != "primary" for s in stage_plan)
    use_parallel = parallel_retrieval_enabled() and len(stage_plan) > 1 and not has_ladder_logic

    if use_parallel:
        _run_retrieval_stages_parallel(
            stage_plan,
            evidence=evidence,
            years_back=years_back,
            pool_max=pool_max,
            merged_arts=merged_arts,
            seen_pm=seen_pm,
            queries=queries,
            stages_executed=stages_executed,
            primary_bundle_holder=primary_holder,
        )
        warn.append(
            "evidence: recuperación paralela (COPILOT_PARALLEL_RETRIEVAL=1); "
            "etapas PubMed/Europe PMC en concurrente."
        )
    else:
        # Bucle único por stage_plan ordenado (Ladder)
        for stage in stage_plan:
            _run_retrieval_stage(
                stage,
                evidence=evidence,
                years_back=years_back,
                pool_max=pool_max,
                merged_arts=merged_arts,
                seen_pm=seen_pm,
                queries=queries,
                stages_executed=stages_executed,
                primary_bundle_holder=primary_holder,
            )
            
            # Cortamos si esta etapa llenó la cuota solicitada
            if (
                stage.stop_after_if_pmids_at_least is not None
                and len(merged_arts) >= stage.stop_after_if_pmids_at_least
            ):
                break

    _maybe_llm_refine_stage(
        evidence=evidence,
        uq=uq,
        ctx_raw=ctx_raw,
        years_back=years_back,
        pool_max=pool_max,
        merged_arts=merged_arts,
        seen_pm=seen_pm,
        queries=queries,
        stages_executed=stages_executed,
        primary_bundle_holder=primary_holder,
        warn=warn,
    )

    expected_lm = evidence_frame.expected_landmark_trials or intent_graph.expected_landmark_trials
    found_lm = landmarks_found_in_articles(
        [a for a in merged_arts if isinstance(a, dict)],
        expected_lm,
    )
    if expected_lm and len(found_lm) < max(1, len(expected_lm) // 3):
        rescue_q = build_landmark_rescue_query_for_graph(intent_graph).strip()
        if rescue_q and rescue_q not in queries:
            rescue_stage = RetrievalStage(
                stage_id="landmark_rescue",
                query=rescue_q,
                years_back_override=20,
                stage_role="supplemental",
                retrieval_tier=2,
            )
            _run_retrieval_stage(
                rescue_stage,
                evidence=evidence,
                years_back=years_back,
                pool_max=pool_max,
                merged_arts=merged_arts,
                seen_pm=seen_pm,
                queries=queries,
                stages_executed=stages_executed,
                primary_bundle_holder=primary_holder,
            )
            warn.append(
                "evidence: rescate landmark activado (landmarks esperados ausentes en pool inicial)."
            )

    primary_bundle = primary_holder[0]

    if not merged_arts:
        warn.append("evidence: 0 artículos recuperados (query o conectividad)")
    bd = (
        primary_bundle.model_dump()
        if primary_bundle is not None
        else {
            "search_term": queries[0] if queries else planned,
            "pmids": [],
            "articles": [],
            "retrieval_debug": {},
        }
    )
    if len(queries) > 1 or adaptive:
        rd = bd.get("retrieval_debug")
        rd = _aggregate_retrieval_debug(rd if isinstance(rd, dict) else {}, stages_executed, merged_arts)
        if isinstance(rd, dict):
            rd["multi_stage_queries"] = queries
            rd["pubmed_pre_rerank_pool_max"] = pool_max
            rd["pubmed_candidates_merged"] = len(merged_arts)
            rd["parallel_retrieval"] = use_parallel
            if adaptive:
                rd["retrieval_stages_planned"] = [
                    {
                        "stage_id": s.stage_id,
                        "stop_after_if_pmids_at_least": s.stop_after_if_pmids_at_least,
                        "run_only_if_pmids_below": s.run_only_if_pmids_below,
                    }
                    for s in retrieval_stages
                ]
                rd["retrieval_stages_executed"] = stages_executed
            bd["retrieval_debug"] = rd
        mode = "paralela" if use_parallel else "multi-etapa"
        warn.append(
            f"evidence: recuperación {mode} ({len(stages_executed)} etapas ejecutadas, "
            f"{len(merged_arts)} candidatos únicos antes de re-ranking)."
        )
    bd["search_term"] = queries[0] if queries else planned
    raw_arts = merged_arts
    clinical_intent_dict: dict[str, Any] | None = None
    if isinstance(raw_arts, list) and raw_arts:
        uq = _graph_effective_user_query(work)
        amin, pconds, pmeds = _cohort_rerank_hints(work)
        intent = evidence_frame.to_clinical_intent()
        clinical_intent_dict = intent.to_dict()
        clinical_ctx = as_clinical_context(work.get("clinical_context"))

        clean_arts = filter_off_topic_abstracts(
            [a for a in raw_arts if isinstance(a, dict)],
            intent,
            intent_graph=intent_graph,
        )
        raw_arts_filtered_count = len(raw_arts) - len(clean_arts)
        if raw_arts_filtered_count > 0:
            warn.append(f"evidence: filtro pre-rerank excluyó {raw_arts_filtered_count} artículos no relacionados al dominio clínico central.")
        
        ranked = rerank_article_dicts(
            clean_arts,
            uq,
            cap=6,
            population_age_min=amin,
            population_conditions=pconds,
            population_medications=pmeds,
            clinical_intent=intent,
            clinical_context=clinical_ctx,
        )
        weak_sem = clinical_weak_evidence_share(
            [
                (str(a.get("title") or ""), str(a.get("abstract_snippet") or ""))
                for a in raw_arts[:12]
                if isinstance(a, dict)
            ],
            clinical_intent=intent,
        )
        bd["articles"] = ranked
        bd["pmids"] = [str(a.get("pmid") or "").strip() for a in ranked if str(a.get("pmid") or "").strip()]
        if len(raw_arts) >= 8:
            warn.append(
                "evidence: se recuperaron varios candidatos; el orden prioriza alineación clínica "
                "(intención PICO), tipo de estudio y cohorte."
            )
        rd = bd.get("retrieval_debug")
        if isinstance(rd, dict):
            rd = dict(rd)
            metrics = rd.get("retrieval_metrics")
            if isinstance(metrics, dict):
                metrics = dict(metrics)
                metrics["clinical_weak_evidence_share_pre_rerank"] = round(weak_sem, 3)
                rd["retrieval_metrics"] = metrics
            sem_dbg = next(
                (a.get("semantic_ranking_debug") for a in ranked if isinstance(a, dict)),
                None,
            )
            if isinstance(sem_dbg, dict):
                rd["semantic_ranking"] = sem_dbg
            bd["retrieval_debug"] = rd
        if isinstance(rd, dict) and (rd.get("synthesis_pubtype_refine") or {}).get("applied"):
            warn.append(
                "evidence: segunda búsqueda PubMed con filtro de tipo (revisión, meta-análisis o ECA) "
                "por alto volumen de resultados o predominio de casos clínicos en la primera página."
            )
    canonical_q, executed_qs = canonical_pubmed_query_after_retrieval(queries)
    out: dict[str, Any] = {
        "evidence_bundle": bd,
        "pubmed_query": canonical_q or planned,
        "pubmed_queries_executed": executed_qs,
        "trace": _new_trace_step(NodeName.EVIDENCE_RETRIEVAL, trace_summary),
        "warnings": warn,
    }
    if len(executed_qs) > 1:
        warn.append(
            f"evidence: pubmed_query canónica = primera etapa ejecutada; "
            f"{len(executed_qs)} queries en pubmed_queries_executed / retrieval_debug."
        )
    if clinical_intent_dict is not None:
        out["clinical_intent"] = clinical_intent_dict
    out["clinical_intent_graph"] = intent_graph.to_dict()
    out["clinical_evidence_frame"] = evidence_frame.to_dict()
    return out


def execute_plan(
    state: CopilotState,
    clinical: ClinicalCapability | None,
    evidence: EvidenceCapability,
    *,
    tool_phase_only: bool = False,
) -> dict[str, Any]:
    route = state["route"]
    plan = build_execution_plan(route)
    work: dict[str, Any] = dict(state)
    acc_trace: list = []
    acc_warn: list = []
    skip_clinical_summary = False
    out: dict[str, Any] = {}

    for step in plan.steps:
        if tool_phase_only and step.kind in ("synthesis", "safety"):
            continue
        delta = _dispatch_step(
            work,
            step,
            route=route,
            clinical=clinical,
            evidence=evidence,
            skip_clinical_summary=skip_clinical_summary,
        )
        if delta is None:
            continue
        if delta.get("_noop"):
            acc_warn.extend(delta.get("warnings") or [])
            continue
        if delta.get("_skip_clinical_summary"):
            skip_clinical_summary = True
            delta = {k: v for k, v in delta.items() if k != "_skip_clinical_summary"}
        _apply_delta(work, delta, acc_trace, acc_warn)
        for k, v in delta.items():
            if k not in ("trace", "warnings"):
                out[k] = v

    return {
        **out,
        "trace": acc_trace,
        "warnings": acc_warn,
    }


def _dispatch_step(
    work: dict[str, Any],
    step: PlanStep,
    *,
    route: Route,
    clinical: ClinicalCapability | None,
    evidence: EvidenceCapability,
    skip_clinical_summary: bool,
) -> dict[str, Any] | None:
    k = step.kind

    if k == "clinical_summary" and skip_clinical_summary:
        return {
            "_noop": True,
            "warnings": [
                "executor: clinical_summary omitido (fusionado en cohort_sql / hybrid_clinical)"
            ],
        }

    if k == "cohort_sql":
        if route == Route.HYBRID:
            d = hybrid_clinical_route_node(work, clinical)
            return {**d, "_skip_clinical_summary": True}
        if route == Route.SQL:
            return sql_route_node(work, clinical)
        return None

    if k == "clinical_summary":
        return None

    if k == "pubmed_query":
        if route == Route.HYBRID:
            return hybrid_pubmed_route_node(work, evidence)
        if route == Route.EVIDENCE:
            return _step_pubmed_evidence_route(work, evidence)
        return None

    if k == "evidence_retrieval":
        if route == Route.HYBRID:
            return _step_evidence_retrieve(
                work,
                evidence,
                trace_summary="EvidenceCapability.retrieve_evidence (hybrid route)",
                years_back=_HYBRID_PUBMED_YEARS_BACK,
            )
        if route == Route.EVIDENCE:
            return _step_evidence_retrieve(
                work,
                evidence,
                trace_summary="EvidenceCapability.retrieve_evidence (evidence route)",
            )
        return None

    if k == "clarify":
        return clarify_node(work)

    if k == "fallback_unknown":
        return unknown_stub_node(work)

    return None
