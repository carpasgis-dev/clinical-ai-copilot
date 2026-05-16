"""
Nodos del grafo LangGraph — router determinista + capabilities inyectables.

La evidencia biomédica pasa siempre por ``EvidenceCapability`` (PubMed real o stub en tests).
``trace`` y ``warnings`` usan reducer ``operator.add``: cada nodo devuelve solo deltas nuevos.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List

from app.capabilities.clinical_sql.cohort_parser import (
    CohortQuery,
    cohort_query_has_filters,
    humanize_like_tokens_es,
    like_tokens_for_display,
    merge_cohort_queries,
    parse_cohort_query,
)
from app.capabilities.clinical_sql.sql_builder import build_sql_from_cohort
from app.capabilities.clinical_sql.terminology import load_cached_terminology
from app.capabilities.contracts import ClinicalCapability, EvidenceCapability
from app.orchestration.llm_synthesis import (
    medical_answer_after_llm_synthesis,
    synthesis_uses_llm,
    try_llm_synthesis_narrative,
)
from app.orchestration.medical_answer_builder import (
    build_stub_medical_answer,
    build_unknown_medical_answer,
    render_medical_answer_to_text,
    sql_cohort_size,
)
from app.orchestration.planner import build_execution_plan, execution_plan_to_jsonable
from app.orchestration.router import classify_route, get_disclaimer
from app.session.memory import (
    clear_pending_clarification,
    load_session_memory,
    save_session_turn_snapshot,
    set_pending_clarification,
    update_session_after_planner,
    update_session_after_sql_route,
)
from app.session.clarification import DEFAULT_CLARIFICATION_QUESTION, parse_clarification_reply
from app.session.followup import is_followup_query
from app.schemas.copilot_state import (
    ClinicalContext,
    CopilotState,
    NodeName,
    Route,
    SqlResult,
    TraceStep,
)


def _new_trace_step(
    node: NodeName,
    summary: str,
    error: str | None = None,
) -> list[TraceStep]:
    """Un solo paso de traza para el reducer append-only."""
    return [TraceStep(node=node, summary=summary, error=error)]


def _graph_effective_user_query(state: CopilotState) -> str:
    """NL efectivo: aclaración fusionada (``resolved_user_query``) o ``user_query``."""
    r = state.get("resolved_user_query")
    if isinstance(r, str) and r.strip():
        return r.strip()
    return (state.get("user_query") or "").strip()


def _persist_sql_session_memory(
    state: CopilotState,
    *,
    cohort_q: CohortQuery,
    sql_result_payload: dict[str, Any] | None,
    structured_cohort_applied: bool,
) -> None:
    """Actualiza memoria RAM si hay ``session_id`` y el SQL no falló."""
    sid = (state.get("session_id") or "").strip()
    if not sid or not sql_result_payload:
        return
    if sql_result_payload.get("error"):
        return
    route = state["route"]
    ex = sql_result_payload.get("executed_query")
    q = str(ex).strip() if ex is not None else ""
    update_session_after_sql_route(
        sid,
        route=route,
        effective_cohort=cohort_q,
        sql_executed=q or None,
        structured_cohort_applied=structured_cohort_applied,
    )


def _try_structured_cohort_sql(
    clinical: ClinicalCapability,
    user_query: str,
    *,
    base_cohort: CohortQuery | None = None,
) -> tuple[SqlResult | None, list[str], str, list[str], CohortQuery]:
    """
    Parser estructurado + SQL de conteo si hay filtros y el esquema lo permite.

    Devuelve ``(SqlResult | None, avisos, texto resumen traza, tables_used, cohort_query)``.
    ``cohort_query`` es la cohorte **efectiva** (parse del turno + fusión con memoria de sesión).
    """
    tables = {t.lower() for t in clinical.list_tables()}
    db_path = (getattr(clinical, "_path", None) or "").strip()
    k_cond, k_med = load_cached_terminology(db_path) if db_path else (frozenset(), frozenset())
    parsed = parse_cohort_query(
        user_query,
        known_condition_terms=k_cond or None,
        known_medication_terms=k_med or None,
    )
    cohort_q = merge_cohort_queries(base_cohort, parsed)
    warns: list[str] = []

    if "patients" not in tables:
        return None, ["sql: base sin tabla patients"], "sin tabla patients", ["patients"], cohort_q

    gtc = getattr(clinical, "get_table_columns", None)
    if not callable(gtc):
        return None, [], "sin get_table_columns", ["patients"], cohort_q

    if not cohort_query_has_filters(cohort_q):
        return None, warns, "sin filtros cohorte estructurados", ["patients"], cohort_q

    pcols = set(gtc("patients"))
    ccols = set(gtc("conditions")) if "conditions" in tables else None
    mcols = set(gtc("medications")) if "medications" in tables else None
    built, wbuild = build_sql_from_cohort(
        cohort_q,
        patient_cols=pcols,
        condition_cols=ccols,
        medication_cols=mcols,
    )
    warns.extend(wbuild)
    trace_detail = "SqliteClinicalCapability cohort structured NL→SQL (conteo filtrado)"
    tables_used = ["patients"]
    if not built:
        warns.append("sql: cohorte filtrada no construida")
        return None, warns, trace_detail, tables_used, cohort_q

    r_try = clinical.run_safe_query(built)
    if r_try.error:
        warns.append(f"sql: consulta filtrada inválida ({r_try.error})")
        return None, warns, trace_detail, tables_used, cohort_q

    if "conditions" in built.lower():
        tables_used.append("conditions")
    if "medications" in built.lower():
        tables_used.append("medications")

    rows = r_try.rows or []
    n = 0
    parsed = False
    if rows and "cohort_size" in rows[0]:
        try:
            n = int(rows[0]["cohort_size"])
            parsed = True
        except (TypeError, ValueError):
            n = 0
            parsed = True
    sql_out = SqlResult(
        executed_query=r_try.executed_query,
        rows=rows,
        row_count=n if parsed else r_try.row_count,
        tables_used=tables_used,
    )
    return sql_out, warns, trace_detail, tables_used, cohort_q


def router_node(state: CopilotState) -> dict[str, Any]:
    """Clasifica la ruta; fusiona turno anterior ambiguo + respuesta corta (fase 3.3)."""
    sid = (state.get("session_id") or "").strip()
    raw_q = (state.get("user_query") or "").strip()
    out_extra: dict[str, Any] = {}

    mem = load_session_memory(sid)
    if sid and mem.pending_clarification and mem.pending_ambiguous_query:
        choice = parse_clarification_reply(raw_q)
        merged = f"{mem.pending_ambiguous_query.strip()}\n{raw_q.strip()}"
        if choice == "sql":
            clear_pending_clarification(sid)
            route = Route.SQL
            route_reason = "post_clarify: usuario eligió datos de cohorte / SQL local"
            out_extra["resolved_user_query"] = merged
        elif choice == "evidence":
            clear_pending_clarification(sid)
            route = Route.EVIDENCE
            route_reason = "post_clarify: usuario eligió evidencia científica / PubMed"
            out_extra["resolved_user_query"] = merged
        elif choice == "hybrid":
            clear_pending_clarification(sid)
            route = Route.HYBRID
            route_reason = "post_clarify: usuario eligió datos de cohorte y evidencia"
            out_extra["resolved_user_query"] = merged
        else:
            route, route_reason = classify_route(raw_q)
            if route != Route.AMBIGUOUS:
                clear_pending_clarification(sid)
    else:
        route, route_reason = classify_route(raw_q)

    out: dict[str, Any] = {
        "route": route,
        "route_reason": route_reason,
        "trace": _new_trace_step(
            NodeName.ROUTER,
            f"route={route.value}; {route_reason}",
        ),
        **out_extra,
    }
    if not state.get("created_at"):
        out["created_at"] = datetime.now(timezone.utc)
    return out


def clarify_node(state: CopilotState) -> dict[str, Any]:
    """
    Fase 3.3 — no ejecuta tools: pide aclaración y guarda la consulta pendiente en sesión RAM.
    """
    sid = (state.get("session_id") or "").strip()
    q = (state.get("user_query") or "").strip()
    if sid:
        set_pending_clarification(sid, q)
    cq = DEFAULT_CLARIFICATION_QUESTION
    body = (
        "Se detectó ambigüedad entre consultar datos de la cohorte local y buscar evidencia científica.\n\n"
        f"{cq}"
    )
    ma = {
        "summary": body,
        "cohort_summary": None,
        "evidence_summary": None,
        "cohort_size": None,
        "key_findings": [],
        "recommendations": [],
        "citations": [],
        "evidence_statements": [],
        "limitations": [
            "Aclaración requerida: el siguiente mensaje debe indicar si desea datos locales, evidencia o ambos.",
        ],
    }
    return {
        "needs_clarification": True,
        "clarification_question": cq,
        "synthesis_draft": render_medical_answer_to_text(ma),
        "final_answer": render_medical_answer_to_text(ma),
        "medical_answer": ma,
        "trace": _new_trace_step(
            NodeName.CLARIFY,
            "needs_clarification=true; awaiting user choice (session RAM)",
        ),
    }


def planner_node(state: CopilotState) -> dict[str, Any]:
    """
    Fase 3.1 — materializa ``ExecutionPlan`` según ``route`` (orquestación explícita).

    El siguiente nodo del grafo (``executor``) ejecuta estos pasos en secuencia.
    """
    route = state["route"]
    plan = build_execution_plan(route)
    ep = execution_plan_to_jsonable(plan)
    kinds = " → ".join(s["kind"] for s in ep)
    sid = (state.get("session_id") or "").strip()
    suffix = ""
    if sid:
        mem = load_session_memory(sid)
        if is_followup_query(state.get("user_query") or "") and mem.last_cohort:
            suffix = " | session: follow-up (cohorte activa en RAM)"
        update_session_after_planner(sid, route, plan)
    return {
        "execution_plan": ep,
        "trace": _new_trace_step(
            NodeName.PLANNER,
            f"ExecutionPlan ({len(ep)} pasos): {kinds}{suffix}",
        ),
    }


def sql_stub_node(state: CopilotState) -> dict[str, Any]:
    """Simula ejecución SQL de solo lectura."""
    sql = SqlResult(
        executed_query="-- stub\nSELECT 1 AS cohort_size",
        rows=[{"cohort_size": 1}],
        row_count=1,
        tables_used=["stub_patients"],
    )
    return {
        "sql_result": sql.model_dump(),
        "trace": _new_trace_step(NodeName.SQL, "stub read-only SQL"),
        "warnings": ["stub: sin I/O real a base de datos"],
    }


def sql_route_node(state: CopilotState, clinical: ClinicalCapability | None) -> dict[str, Any]:
    """Ruta SQL: stub si no hay BD; si no, conteo de cohorte (NL heurístico → SQL o conteo simple)."""
    if clinical is None:
        return sql_stub_node(state)

    tables = {t.lower() for t in clinical.list_tables()}
    if "patients" not in tables:
        return {
            "sql_result": SqlResult(
                executed_query="",
                error="no hay tabla patients",
            ).model_dump(),
            "trace": _new_trace_step(
                NodeName.SQL,
                "SqliteClinicalCapability: sin tabla patients",
                error="no hay tabla patients",
            ),
            "warnings": ["sql: base sin tabla patients"],
        }

    mem = load_session_memory(state.get("session_id") or "")
    uq = _graph_effective_user_query(state)
    base = mem.last_cohort if (mem.last_cohort is not None and is_followup_query(uq)) else None
    r, extra_warns, trace_detail, tables_used, cohort_q = _try_structured_cohort_sql(
        clinical, uq, base_cohort=base
    )
    structured_used = r is not None
    if r is None:
        trace_detail = "SqliteClinicalCapability cohort vivos (conteo simple)"
        tables_used = ["patients"]

    if r is None:
        q_alive = (
            "SELECT COUNT(*) AS cohort_size FROM patients "
            "WHERE COALESCE(TRIM(deathdate), '') = ''"
        )
        r = clinical.run_safe_query(q_alive)
        if r.error:
            r = clinical.run_safe_query("SELECT COUNT(*) AS cohort_size FROM patients")
        if r.error:
            empty = SqlResult(executed_query=q_alive, error=r.error)
            return {
                "sql_result": empty.model_dump(),
                "trace": _new_trace_step(
                    NodeName.SQL,
                    "SqliteClinicalCapability cohort query falló",
                    error=r.error,
                ),
                "warnings": [f"sql: {r.error}", *extra_warns],
            }

    rows = r.rows or []

    n = 0
    parsed = False
    if rows and "cohort_size" in rows[0]:
        try:
            n = int(rows[0]["cohort_size"])
            parsed = True
        except (TypeError, ValueError):
            n = 0
            parsed = True
    sql_out = SqlResult(
        executed_query=r.executed_query,
        rows=rows,
        row_count=n if parsed else r.row_count,
        tables_used=tables_used,
    )
    payload = sql_out.model_dump()
    _persist_sql_session_memory(
        state,
        cohort_q=cohort_q,
        sql_result_payload=payload,
        structured_cohort_applied=structured_used,
    )
    return {
        "sql_result": payload,
        "trace": _new_trace_step(NodeName.SQL, f"{trace_detail} n≈{sql_out.row_count}"),
        "warnings": extra_warns,
    }


def hybrid_clinical_stub_node(state: CopilotState) -> dict[str, Any]:
    """Simula extracción de perfil clínico para ruta HYBRID (Capability A pendiente)."""
    ctx = ClinicalContext(
        age_range=">65",
        conditions=["diabetes mellitus tipo 2", "hipertensión arterial"],
        medications=["metformina"],
        population_hint="stub cohort profile",
        population_conditions=["diabet"],
        population_medications=["metform"],
        population_age_min=65,
        population_size=12,
    )
    return {
        "clinical_context": ctx.model_dump(),
        "trace": _new_trace_step(
            NodeName.CLINICAL_SUMMARY,
            "stub structured clinical summary",
        ),
    }


def hybrid_clinical_route_node(
    state: CopilotState,
    clinical: ClinicalCapability | None,
) -> dict[str, Any]:
    """
    Híbrido: cohorte SQL real (mismo pipeline que ruta SQL) + resumen clínico para PubMed.

    ``sql_result`` queda en el estado cuando el parse estructurado dispara un conteo válido;
    ``clinical_context`` se enriquece con ``population_size`` si hay ``cohort_size``.
    """
    if clinical is None:
        return hybrid_clinical_stub_node(state)

    mem = load_session_memory(state.get("session_id") or "")
    uq = _graph_effective_user_query(state)
    base = mem.last_cohort if (mem.last_cohort is not None and is_followup_query(uq)) else None
    sql_r, sql_warns, sql_trace, _tables_used, cohort_q = _try_structured_cohort_sql(
        clinical, uq, base_cohort=base
    )
    ctx = clinical.extract_clinical_summary(uq)
    warn: List[str] = list(sql_warns)

    pop_updates: dict[str, Any] = {
        "population_conditions": humanize_like_tokens_es(
            like_tokens_for_display(cohort_q.condition_like_tokens)
        ),
        "population_medications": humanize_like_tokens_es(
            like_tokens_for_display(cohort_q.medication_like_tokens)
        ),
        "population_age_min": cohort_q.age_min_years,
        "population_age_max": cohort_q.age_max_years,
        "population_sex": cohort_q.sex,
    }
    if cohort_q.age_min_years is not None:
        pop_updates["age_range"] = f">={cohort_q.age_min_years}"
    elif cohort_q.age_max_years is not None:
        pop_updates["age_range"] = f"<{cohort_q.age_max_years}"

    if sql_r is not None:
        n = sql_cohort_size(sql_r.model_dump())
        if n is not None:
            hint = ctx.population_hint or "Perfil agregado desde BD local"
            pop_updates["population_size"] = n
            pop_updates["population_hint"] = f"{hint} — conteo cohorte acotada a la consulta"

    ctx = ctx.model_copy(update=pop_updates)

    if (
        not ctx.conditions
        and not ctx.medications
        and not ctx.population_conditions
        and not ctx.population_medications
        and ctx.population_size in (None, 0)
    ):
        warn.append("clinical: cohorte sin condiciones/medicamentos detectables en BD")

    trace_parts = ["SqliteClinicalCapability.extract_clinical_summary"]
    if sql_r is not None:
        trace_parts.insert(
            0,
            f"hybrid: {sql_trace} n≈{sql_r.row_count}",
        )
    out: dict[str, Any] = {
        "clinical_context": ctx.model_dump(),
        "trace": _new_trace_step(
            NodeName.CLINICAL_SUMMARY,
            "; ".join(trace_parts),
        ),
        "warnings": warn,
    }
    if sql_r is not None:
        sql_payload = sql_r.model_dump()
        out["sql_result"] = sql_payload
        _persist_sql_session_memory(
            state,
            cohort_q=cohort_q,
            sql_result_payload=sql_payload,
            structured_cohort_applied=True,
        )
    return out


def hybrid_pubmed_route_node(state: CopilotState, evidence: EvidenceCapability) -> dict[str, Any]:
    """Vista previa heurística; la query canónica se fija tras ``evidence_retrieval``."""
    from app.capabilities.evidence_rag.heuristic_evidence_query import preview_pubmed_query

    built = preview_pubmed_query(
        _graph_effective_user_query(state),
        state.get("clinical_context"),
    )
    warn: List[str] = []
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


def unknown_stub_node(state: CopilotState) -> dict[str, Any]:
    """Mensaje mínimo cuando el router no encuentra señales claras."""
    msg = (
        "No se pudo clasificar la consulta con las señales actuales (stub). "
        "Reformule o añada contexto clínico o de evidencia."
    )
    ma = build_unknown_medical_answer(msg)
    draft = render_medical_answer_to_text(ma)
    return {
        "synthesis_draft": draft,
        "final_answer": draft,
        "medical_answer": ma,
        "trace": _new_trace_step(
            NodeName.FALLBACK,
            "stub unknown-route handler",
        ),
    }


def synthesis_stub_node(state: CopilotState) -> dict[str, Any]:
    """
    Construye ``MedicalAnswer`` determinista (fuente estructurada en API).

    Con ``COPILOT_SYNTHESIS=llm``, ``final_answer`` es solo la narrativa del LLM si tiene éxito;
    si falla, se usa el render determinista de ``MedicalAnswer``.
    """
    ma = build_stub_medical_answer(state)
    draft = render_medical_answer_to_text(ma)
    trace_summary = "deterministic MedicalAnswer (structured source of truth)"
    extra_warns: list[str] = []

    llm_text, llm_warns = try_llm_synthesis_narrative(state, dict(ma))
    extra_warns.extend(llm_warns)
    out_ma = dict(ma)
    final_body = draft
    if llm_text:
        final_body = llm_text
        out_ma = medical_answer_after_llm_synthesis(out_ma)
        trace_summary = "deterministic MedicalAnswer + LLM narrative (OpenAI-compatible /v1)"
    elif synthesis_uses_llm() and llm_warns:
        w0 = str(llm_warns[0]).strip()
        if len(w0) > 180:
            w0 = w0[:177] + "…"
        trace_summary = f"deterministic MedicalAnswer (LLM synthesis no aplicada: {w0})"

    return {
        "synthesis_draft": final_body,
        "final_answer": final_body,
        "medical_answer": out_ma,
        "trace": _new_trace_step(NodeName.SYNTHESIS, trace_summary),
        "warnings": extra_warns,
    }


def safety_node(state: CopilotState) -> dict[str, Any]:
    """Aplica disclaimer estático y deja trazabilidad del nodo Safety."""
    route = state["route"]
    disclaimer = get_disclaimer(route)
    body = state.get("final_answer") or ""
    final = f"{body.rstrip()}\n\n---\n{disclaimer}"
    out: dict[str, Any] = {
        "disclaimer": disclaimer,
        "final_answer": final,
        "trace": _new_trace_step(NodeName.SAFETY, "disclaimer applied"),
    }
    ma_raw = state.get("medical_answer")
    if isinstance(ma_raw, dict):
        lim = list(ma_raw.get("limitations") or [])
        d = str(disclaimer).strip()
        if d and not any(d == str(x).strip() for x in lim):
            lim.append(d)
        out["medical_answer"] = {**ma_raw, "limitations": lim}
    sid = (state.get("session_id") or "").strip()
    if sid:
        ma_save = out.get("medical_answer")
        if not isinstance(ma_save, dict):
            ma_save = state.get("medical_answer") if isinstance(state.get("medical_answer"), dict) else None
        save_session_turn_snapshot(
            sid,
            user_query=str(state.get("user_query") or ""),
            sql_result=state.get("sql_result") if isinstance(state.get("sql_result"), dict) else None,
            medical_answer=ma_save,
        )
    return out


def executor_node(
    state: CopilotState,
    clinical: ClinicalCapability | None,
    evidence: EvidenceCapability,
) -> dict[str, Any]:
    """Fase 3.6 — ejecuta solo la fase de herramientas del ``ExecutionPlan`` (sin síntesis ni safety)."""
    from app.orchestration.executor import execute_plan

    return execute_plan(state, clinical, evidence, tool_phase_only=True)


def reasoning_node(state: CopilotState) -> dict[str, Any]:
    """Fase 3.7.a — ``ReasoningState`` solo cuando hay cohorte/evidencia materializada en el turno."""
    from app.orchestration.reasoning import build_reasoning_state, reasoning_state_to_dict

    route = state.get("route")
    if isinstance(route, str):
        try:
            route = Route(route)
        except ValueError:
            route = Route.UNKNOWN

    if route in (Route.UNKNOWN, Route.AMBIGUOUS):
        return {"reasoning_state": None}

    rs = build_reasoning_state(dict(state))
    rs_dict = reasoning_state_to_dict(rs)
    return {
        "reasoning_state": rs_dict,
        "trace": _new_trace_step(
            NodeName.REASONING,
            "ReasoningState determinista (cohorte + evidencia + aplicabilidad)",
        ),
    }
