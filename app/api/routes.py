"""Rutas HTTP."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.api.clinical_factory import clinical_cache_token
from app.api.evidence_factory import resolve_evidence_backend
from app.api.graph_cache import get_compiled_graph
from app.capabilities.evidence_rag.copilot_errors import CopilotError
from app.capabilities.evidence_rag.query_planning import query_planner_cache_token
from app.api.response_format import build_query_response
from app.orchestration.medical_answer_builder import citations_from_state
from app.api.schemas import HERO_QUERY_EXAMPLE, Http422Response, QueryRequest, QueryResponse
from app.eval.run_logger import log_eval_event

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        422: {
            "model": Http422Response,
            "description": (
                "Cuerpo JSON inválido o que no cumple ``QueryRequest``: "
                "falta el campo ``query``, ``query`` vacío (mínimo 1 carácter), tipos incorrectos, etc. "
                "Véase el schema **Http422Response** (estructura de ``detail``)."
            ),
            "content": {
                "application/json": {
                    "examples": {
                        "campo_query_ausente": {
                            "summary": "Campo query ausente",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "query"],
                                        "msg": "Field required",
                                        "type": "missing",
                                        "input": None,
                                        "ctx": None,
                                        "url": None,
                                    }
                                ]
                            },
                        },
                        "query_vacio": {
                            "summary": "Campo query vacío (min_length=1)",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "query"],
                                        "msg": "String should have at least 1 character",
                                        "type": "string_too_short",
                                        "input": "",
                                        "ctx": {"min_length": 1},
                                        "url": None,
                                    }
                                ]
                            },
                        },
                        "json_invalido": {
                            "summary": "JSON malformado en el cuerpo",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", 0],
                                        "msg": "JSON decode error",
                                        "type": "json_invalid",
                                        "input": None,
                                        "ctx": None,
                                        "url": None,
                                    }
                                ]
                            },
                        },
                    }
                }
            },
        },
    },
    summary="Ejecutar consulta clínica (grafo LangGraph)",
    description=(
        "Ejecuta el pipeline completo (router → planner → executor → reasoning → síntesis → safety). "
        "**422:** cuerpo JSON inválido, falta el campo `query`, o `query` vacío (ver ``QueryRequest`` y schema **Http422Response** en esta misma operación). "
        "Para el **caso hero** (híbrido + PubMed real), envía el ejemplo de `query` del schema. "
        "Configura `COPILOT_EVIDENCE_BACKEND=ncbi` (default) y `NCBI_EMAIL` para mejores cuotas. "
        "Opcional: `COPILOT_SYNTHESIS=llm` + `LLM_BASE_URL` / `LLM_MODEL` (p. ej. llama.cpp u Ollama; si la URL es solo `host:puerto`, se añade `/v1`) para redactar `final_answer` con LLM; "
        "`COPILOT_SYNTHESIS_TIMEOUT` (default 90 s) evita cortes en modelos lentos; si falla, se usa la síntesis determinista.\n\n"
        "**Sobre el schema 200 en Swagger:** *Example value* refleja tipos (p. ej. ``string``); "
        "el JSON real lo devuelve *Execute*. Despliega ``QueryResponse`` en *Schemas* para ver campos anidados."
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
    except CopilotError as exc:
        failure_kind = exc.code
        err_msg = exc.message
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
        status = 504 if exc.code == "PUBMED_TIMEOUT" else 502
        raise HTTPException(
            status_code=status,
            detail={"error_code": exc.code, "message": exc.message},
        ) from exc
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
