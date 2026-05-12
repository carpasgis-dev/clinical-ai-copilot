"""Convierte ``CopilotState`` final en ``QueryResponse`` + extracción de citas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.api.schemas import CitationOut, QueryResponse, SqlResultOut, TraceStepOut
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


def _evidence_bundle_dict(state: Dict[str, Any]) -> Optional[dict[str, Any]]:
    eb = state.get("evidence_bundle")
    if eb is None:
        return None
    if isinstance(eb, dict):
        return eb
    if hasattr(eb, "model_dump"):
        return eb.model_dump(mode="json")
    return None


def citations_from_state(state: Dict[str, Any]) -> tuple[List[str], List[CitationOut]]:
    eb = _evidence_bundle_dict(state)
    if not eb:
        return [], []
    pmids_top = [str(p) for p in (eb.get("pmids") or [])]
    cites: List[CitationOut] = []
    for art in eb.get("articles") or []:
        if not isinstance(art, dict):
            continue
        pmid = str(art.get("pmid") or "").strip()
        if not pmid:
            continue
        title = str(art.get("title") or "")
        year = art.get("year")
        doi = art.get("doi")
        y_int: Optional[int] = None
        if year is not None and str(year).strip().isdigit():
            y_int = int(str(year)[:4])
        cites.append(
            CitationOut(
                pmid=pmid,
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                year=y_int,
                doi=str(doi) if doi else None,
            )
        )
    pmids = pmids_top or [c.pmid for c in cites]
    return pmids, cites


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
    preview = rows_in[:_SQL_PREVIEW_ROWS]
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
        pubmed_query=state.get("pubmed_query"),
        sql_result=_sql_result_out(state),
        final_answer=str(state.get("final_answer") or ""),
        disclaimer=str(state.get("disclaimer") or ""),
        trace=_trace_to_steps(state.get("trace")),
        pmids=pmids,
        citations=citations,
        session_id=session_id,
        latency_ms=round(latency_ms, 3),
        ok=ok,
        error=error,
    )
