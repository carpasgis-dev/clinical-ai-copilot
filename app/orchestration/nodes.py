"""
Nodos del grafo LangGraph — router determinista + capabilities inyectables.

La evidencia biomédica pasa siempre por ``EvidenceCapability`` (PubMed real o stub en tests).
``trace`` y ``warnings`` usan reducer ``operator.add``: cada nodo devuelve solo deltas nuevos.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List

from app.capabilities.contracts import ClinicalCapability, EvidenceCapability
from app.orchestration.router import classify_route, get_disclaimer
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


def router_node(state: CopilotState) -> dict[str, Any]:
    """Clasifica la ruta y deja trazabilidad del router."""
    query = state["user_query"]
    route, route_reason = classify_route(query)
    out: dict[str, Any] = {
        "route": route,
        "route_reason": route_reason,
        "trace": _new_trace_step(
            NodeName.ROUTER,
            f"route={route.value}; {route_reason}",
        ),
    }
    if not state.get("created_at"):
        out["created_at"] = datetime.now(timezone.utc)
    return out


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
    """Ruta SQL: stub si no hay BD; si no, conteo de cohorte viva vía ``SqliteClinicalCapability``."""
    if clinical is None:
        return sql_stub_node(state)
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
            "warnings": [f"sql: {r.error}"],
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
        tables_used=["patients"],
    )
    return {
        "sql_result": sql_out.model_dump(),
        "trace": _new_trace_step(
            NodeName.SQL,
            f"SqliteClinicalCapability cohort vivos n≈{sql_out.row_count}",
        ),
        "warnings": [],
    }


def evidence_route_node(state: CopilotState, evidence: EvidenceCapability) -> dict[str, Any]:
    """Ruta EVIDENCE: ``build_pubmed_query`` + ``retrieve_evidence`` sin contexto clínico."""
    query = state["user_query"]
    built = evidence.build_pubmed_query(query, None)
    bundle = evidence.retrieve_evidence(built, retmax=6, years_back=5)
    warn: List[str] = []
    if not bundle.articles:
        warn.append("evidence: 0 artículos recuperados (query o conectividad)")
    return {
        "evidence_bundle": bundle.model_dump(),
        "trace": _new_trace_step(
            NodeName.EVIDENCE_RETRIEVAL,
            "EvidenceCapability.retrieve_evidence (evidence route)",
        ),
        "warnings": warn,
    }


def hybrid_clinical_stub_node(state: CopilotState) -> dict[str, Any]:
    """Simula extracción de perfil clínico para ruta HYBRID (Capability A pendiente)."""
    ctx = ClinicalContext(
        age_range=">65",
        conditions=["diabetes mellitus tipo 2", "hipertensión arterial"],
        medications=["metformina"],
        population_hint="stub cohort profile",
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
    """Híbrido: resumen desde SQLite (ETL Synthea) o stub si no hay capability clínica."""
    if clinical is None:
        return hybrid_clinical_stub_node(state)
    ctx = clinical.extract_clinical_summary(state["user_query"])
    warn: List[str] = []
    if not ctx.conditions and not ctx.medications and ctx.population_size in (None, 0):
        warn.append("clinical: cohorte sin condiciones/medicamentos detectables en BD")
    return {
        "clinical_context": ctx.model_dump(),
        "trace": _new_trace_step(
            NodeName.CLINICAL_SUMMARY,
            "SqliteClinicalCapability.extract_clinical_summary",
        ),
        "warnings": warn,
    }


def hybrid_pubmed_route_node(state: CopilotState, evidence: EvidenceCapability) -> dict[str, Any]:
    """Construye la query PubMed vía ``EvidenceCapability``."""
    built = evidence.build_pubmed_query(
        state["user_query"],
        state.get("clinical_context"),
    )
    warn: List[str] = []
    if not (built or "").strip():
        warn.append("evidence: build_pubmed_query devolvió cadena vacía")
    return {
        "pubmed_query": built,
        "trace": _new_trace_step(
            NodeName.PUBMED_QUERY_BUILDER,
            "EvidenceCapability.build_pubmed_query",
        ),
        "warnings": warn,
    }


def hybrid_evidence_route_node(state: CopilotState, evidence: EvidenceCapability) -> dict[str, Any]:
    """Recupera evidencia para la query ya construida en ``pubmed_query``."""
    pq = (state.get("pubmed_query") or state["user_query"] or "").strip()
    bundle = evidence.retrieve_evidence(pq, retmax=6, years_back=5)
    warn: List[str] = []
    if not bundle.articles:
        warn.append("evidence: 0 artículos recuperados (query o conectividad)")
    return {
        "evidence_bundle": bundle.model_dump(),
        "trace": _new_trace_step(
            NodeName.EVIDENCE_RETRIEVAL,
            "EvidenceCapability.retrieve_evidence (hybrid route)",
        ),
        "warnings": warn,
    }


def unknown_stub_node(state: CopilotState) -> dict[str, Any]:
    """Mensaje mínimo cuando el router no encuentra señales claras."""
    msg = (
        "No se pudo clasificar la consulta con las señales actuales (stub). "
        "Reformule o añada contexto clínico o de evidencia."
    )
    return {
        "synthesis_draft": msg,
        "final_answer": msg,
        "trace": _new_trace_step(
            NodeName.FALLBACK,
            "stub unknown-route handler",
        ),
    }


def synthesis_stub_node(state: CopilotState) -> dict[str, Any]:
    """Ensambla un borrador legible a partir de los campos presentes."""
    route = state["route"]
    lines: List[str] = [
        "[synthesis stub]",
        f"session_id={state.get('session_id', '')}",
        f"route={route.value}",
        f"query={state['user_query']!r}",
    ]
    if state.get("sql_result"):
        lines.append("capability=sql_result present")
    if state.get("evidence_bundle"):
        lines.append("capability=evidence_bundle present")
    if state.get("clinical_context"):
        lines.append("capability=clinical_context present")
    if state.get("pubmed_query"):
        lines.append(f"pubmed_query={state['pubmed_query']!r}")

    draft = "\n".join(lines)
    return {
        "synthesis_draft": draft,
        "final_answer": draft,
        "trace": _new_trace_step(NodeName.SYNTHESIS, "stub synthesis"),
    }


def safety_node(state: CopilotState) -> dict[str, Any]:
    """Aplica disclaimer estático y deja trazabilidad del nodo Safety."""
    route = state["route"]
    disclaimer = get_disclaimer(route)
    body = state.get("final_answer") or ""
    final = f"{body.rstrip()}\n\n---\n{disclaimer}"
    return {
        "disclaimer": disclaimer,
        "final_answer": final,
        "trace": _new_trace_step(NodeName.SAFETY, "disclaimer applied"),
    }
