"""Convierte ``CopilotState`` final en ``QueryResponse`` + extracción de citas."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.api.schemas import (
    MedicalAnswer,
    PlanStepOut,
    PubMedRetrievalStatusOut,
    QueryResponse,
    ReasoningStateOut,
    SqlPreviewRow,
    SqlResultOut,
    TraceStepOut,
)
from app.orchestration.medical_answer_builder import citations_from_state
from app.schemas.copilot_state import Route, TraceStep


def _started_at_to_str(v: Any) -> str:
    if isinstance(v, datetime):
        return v.isoformat().replace("+00:00", "Z")
    return str(v or "")


def _trace_to_jsonable(trace: Any) -> List[dict[str, Any]]:
    if not trace:
        return []
    out: List[dict[str, Any]] = []
    for step in trace:
        if isinstance(step, TraceStep):
            out.append(step.model_dump(mode="json"))
        elif isinstance(step, dict):
            out.append(step)
        else:
            out.append({"raw": repr(step)})
    return out


def _trace_to_steps(trace: Any) -> List[TraceStepOut]:
    """Convierte trazas del estado a modelos tipados para la respuesta HTTP."""
    raw = _trace_to_jsonable(trace)
    out: List[TraceStepOut] = []
    for d in raw:
        if not isinstance(d, dict):
            continue
        node = d.get("node")
        if hasattr(node, "value"):
            node_s = str(node.value)
        else:
            node_s = str(node or "")
        err = d.get("error")
        out.append(
            TraceStepOut(
                node=node_s,
                started_at=_started_at_to_str(d.get("started_at")),
                duration_ms=d.get("duration_ms"),
                summary=str(d.get("summary") or ""),
                error=None if err is None else str(err),
            )
        )
    return out


def _route_value(route: Any) -> str:
    if isinstance(route, Route):
        return route.value
    return str(route)


def _execution_plan_out(state: Dict[str, Any]) -> List[PlanStepOut]:
    raw = state.get("execution_plan")
    if not raw or not isinstance(raw, list):
        return []
    out: List[PlanStepOut] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            PlanStepOut(
                kind=str(item.get("kind") or ""),
                reason=None if item.get("reason") is None else str(item["reason"]),
            )
        )
    return out


def _evidence_bundle_dict(state: Dict[str, Any]) -> Optional[dict[str, Any]]:
    eb = state.get("evidence_bundle")
    if eb is None:
        return None
    if isinstance(eb, dict):
        return eb
    if hasattr(eb, "model_dump"):
        return eb.model_dump(mode="json")
    return None


def _pubmed_url_out(state: Dict[str, Any]) -> Optional[str]:
    """URL de búsqueda web; prioriza la query normalizada enviada a ESearch si existe."""
    term: Optional[str] = None
    eb = _evidence_bundle_dict(state)
    if eb:
        rd = eb.get("retrieval_debug")
        if isinstance(rd, dict):
            nq = str(rd.get("normalized_query") or "").strip()
            if nq:
                term = nq
    if not term:
        pq = state.get("pubmed_query")
        if pq is None or not str(pq).strip():
            return None
        term = str(pq).strip()
    from app.capabilities.evidence_rag.ncbi.pubmed_urls import pubmed_web_search_url

    return pubmed_web_search_url(term)


def _reasoning_state_out(state: Dict[str, Any]) -> Optional[ReasoningStateOut]:
    raw = state.get("reasoning_state")
    if not isinstance(raw, dict):
        return None
    try:
        return ReasoningStateOut.model_validate(raw)
    except ValidationError:
        return None


def _pubmed_retrieval_status_out(state: Dict[str, Any]) -> Optional[PubMedRetrievalStatusOut]:
    eb = _evidence_bundle_dict(state)
    if not eb:
        return None
    rd = eb.get("retrieval_debug")
    if not isinstance(rd, dict):
        return None
    try:
        return PubMedRetrievalStatusOut.model_validate(rd)
    except ValidationError:
        return None


_SQL_PREVIEW_ROWS = 5


def _sql_result_out(state: Dict[str, Any]) -> Optional[SqlResultOut]:
    """Extrae ``sql_result`` del estado para la API (muestra de filas acotada)."""
    raw = state.get("sql_result")
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        d = raw.model_dump(mode="json")
    elif isinstance(raw, dict):
        d = dict(raw)
    else:
        return None
    rows_in = d.get("rows") or []
    if not isinstance(rows_in, list):
        rows_in = []
    preview_raw = rows_in[:_SQL_PREVIEW_ROWS]
    preview: List[SqlPreviewRow] = []
    for r in preview_raw:
        if isinstance(r, dict):
            try:
                preview.append(SqlPreviewRow.model_validate(r))
            except ValidationError:
                preview.append(
                    SqlPreviewRow.model_validate(
                        json.loads(json.dumps(r, default=str)),
                    )
                )
        else:
            preview.append(SqlPreviewRow.model_validate({}))
    rc = d.get("row_count")
    try:
        row_count = int(rc) if rc is not None else 0
    except (TypeError, ValueError):
        row_count = 0
    return SqlResultOut(
        executed_query=str(d.get("executed_query") or ""),
        row_count=row_count,
        tables_used=list(d.get("tables_used") or []),
        error=None if d.get("error") is None else str(d.get("error")),
        rows=preview,
    )


def _medical_answer_out(state: Dict[str, Any]) -> Optional[MedicalAnswer]:
    raw = state.get("medical_answer")
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return MedicalAnswer.model_validate(raw)
    except Exception:
        return None


def build_query_response(
    state: Dict[str, Any],
    *,
    session_id: str,
    latency_ms: float,
    ok: bool = True,
    error: Optional[str] = None,
) -> QueryResponse:
    pmids, citations = citations_from_state(state)
    return QueryResponse(
        route=_route_value(state.get("route")),
        route_reason=str(state.get("route_reason") or ""),
        execution_plan=_execution_plan_out(state),
        reasoning_state=_reasoning_state_out(state),
        pubmed_query=state.get("pubmed_query"),
        pubmed_queries_executed=state.get("pubmed_queries_executed"),
        pubmed_url=_pubmed_url_out(state),
        pubmed_retrieval_status=_pubmed_retrieval_status_out(state),
        sql_result=_sql_result_out(state),
        final_answer=str(state.get("final_answer") or ""),
        medical_answer=_medical_answer_out(state),
        disclaimer=str(state.get("disclaimer") or ""),
        trace=_trace_to_steps(state.get("trace")),
        pmids=pmids,
        citations=citations,
        needs_clarification=bool(state.get("needs_clarification")),
        clarification_question=(
            None
            if not state.get("needs_clarification")
            or state.get("clarification_question") in (None, "")
            else str(state.get("clarification_question"))
        ),
        session_id=session_id,
        latency_ms=round(latency_ms, 3),
        ok=ok,
        error=error,
    )
