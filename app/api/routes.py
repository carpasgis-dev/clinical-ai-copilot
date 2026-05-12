"""Rutas HTTP."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.api.clinical_factory import clinical_cache_token
from app.api.evidence_factory import resolve_evidence_backend
from app.api.graph_cache import get_compiled_graph
from app.capabilities.evidence_rag.query_planning import query_planner_cache_token
from app.api.response_format import build_query_response, citations_from_state
from app.api.schemas import HERO_QUERY_EXAMPLE, QueryRequest, QueryResponse
from app.eval.run_logger import log_eval_event

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ejecutar consulta clínica (grafo LangGraph)",
    description=(
        "Ejecuta el pipeline completo (router → capabilities → síntesis stub → safety). "
        "Para el **caso hero** (híbrido + PubMed real), envía el ejemplo de `query` del schema. "
        "Configura `COPILOT_EVIDENCE_BACKEND=ncbi` (default) y `NCBI_EMAIL` para mejores cuotas.\n\n"
        "**Sobre el schema 200 en Swagger:** *Example value* es **solo ilustrativo** (tipos → "
        "``string``, ``0``, etc.). **No** indica que `final_answer` vaya vacío: el cuerpo real lo "
        "devuelve *Execute*. En *Schemas*, despliega `QueryResponse` → `TraceStepOut` para ver "
        "los campos de `trace`."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "hero_hybrid": {
                            "summary": "Caso hero (HYBRID + evidencia)",
                            "value": {
                                "query": HERO_QUERY_EXAMPLE,
                                "session_id": "demo-hero-1",
                            },
                        }
                    }
                }
            }
        }
    },
)
def post_query(body: QueryRequest) -> QueryResponse:
    backend = resolve_evidence_backend()
    graph = get_compiled_graph(
        backend,
        query_planner_cache_token(),
        clinical_cache_token(),
    )
    session_id = (body.session_id or "").strip() or str(uuid.uuid4())
    t0 = time.perf_counter()
    failure_kind: str | None = None
    err_msg: str | None = None
    raw: Dict[str, Any] = {}

    try:
        raw = graph.invoke({"user_query": body.query.strip(), "session_id": session_id})
    except Exception as exc:  # noqa: BLE001 — API: log + respuesta controlada
        failure_kind = "invoke_error"
        err_msg = str(exc)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        log_eval_event(
            {
                "session_id": session_id,
                "query": body.query,
                "route": None,
                "route_reason": None,
                "pmids": [],
                "latency_ms": round(latency_ms, 3),
                "ok": False,
                "error": err_msg,
                "failure_kind": failure_kind,
            }
        )
        raise HTTPException(status_code=500, detail=err_msg) from exc

    latency_ms = (time.perf_counter() - t0) * 1000.0
    pmids, _ = citations_from_state(raw)
    log_eval_event(
        {
            "session_id": session_id,
            "query": body.query,
            "route": raw.get("route"),
            "route_reason": raw.get("route_reason"),
            "pmids": pmids,
            "latency_ms": round(latency_ms, 3),
            "ok": True,
            "error": None,
            "failure_kind": None,
        }
    )
    return build_query_response(
        raw,
        session_id=session_id,
        latency_ms=latency_ms,
        ok=True,
        error=None,
    )
