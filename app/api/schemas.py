"""Esquemas Pydantic para ``POST /query``."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

_SQL_PREVIEW_MAX_ROWS = 5


HERO_QUERY_EXAMPLE = (
    "Paciente con diabetes e hipertensión mayor de 65 años. "
    "¿Qué evidencia reciente existe sobre tratamientos que "
    "reduzcan riesgo cardiovascular?"
)


class QueryRequest(BaseModel):
    """Cuerpo de ``POST /query``."""

    query: str = Field(
        ...,
        min_length=1,
        description="Consulta en lenguaje natural.",
        json_schema_extra={"examples": [HERO_QUERY_EXAMPLE]},
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Id de sesión para trazas y logs; si se omite se genera uno.",
    )


class CitationOut(BaseModel):
    """Cita verificable hacia PubMed."""

    pmid: str
    title: str
    url: str
    year: Optional[int] = None
    doi: Optional[str] = None


class SqlResultOut(BaseModel):
    """Fragmento del resultado SQL para auditoría (no vuelca tablas completas)."""

    executed_query: str = Field(
        "",
        description="Sentencia SELECT (o WITH) ejecutada en la capability clínica.",
    )
    row_count: int = Field(0, description="Número de filas devuelto o conteo agregado según nodo SQL.")
    tables_used: List[str] = Field(
        default_factory=list,
        description="Tablas referidas cuando el backend las informa.",
    )
    error: Optional[str] = Field(None, description="Error de ejecución SQL, si lo hubo.")
    rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=f"Hasta {_SQL_PREVIEW_MAX_ROWS} filas de muestra (p. ej. conteo agregado).",
    )


class TraceStepOut(BaseModel):
    """Un paso del grafo en la respuesta HTTP (tipado para OpenAPI / Swagger)."""

    node: str = Field(..., description="Nombre del nodo LangGraph (router, synthesis, …).")
    started_at: str = Field(..., description="Marca de tiempo ISO 8601 (UTC).")
    duration_ms: Optional[float] = Field(
        None, description="Duración del paso en milisegundos, si se midió."
    )
    summary: str = Field("", description="Resumen legible para auditoría.")
    error: Optional[str] = Field(None, description="Mensaje de error del nodo, si hubo fallo.")


class QueryResponse(BaseModel):
    """Respuesta estructurada del copiloto (E2E / UI)."""

    route: str = Field(..., description="Ruta elegida por el router (sql|evidence|hybrid|unknown).")
    route_reason: str = Field(..., description="Explicación breve de por qué se eligió la ruta.")
    pubmed_query: Optional[str] = Field(
        None,
        description="Término enviado a PubMed / Europe PMC; null si no aplica.",
    )
    sql_result: Optional[SqlResultOut] = Field(
        None,
        description="SQL ejecutado en ruta `sql` (y muestra de filas); null en evidence/hybrid/unknown.",
    )
    final_answer: str = Field(
        ...,
        description="Texto tras síntesis (hoy stub; sustituible por LLM).",
    )
    disclaimer: str = Field(..., description="Aviso del nodo safety.")
    trace: List[TraceStepOut] = Field(
        default_factory=list,
        description="Pasos ejecutados para auditoría.",
    )
    pmids: List[str] = Field(default_factory=list, description="PMIDs devueltos por la búsqueda.")
    citations: List[CitationOut] = Field(
        default_factory=list,
        description="Metadatos por cita (alineado con pmids).",
    )
    session_id: str
    latency_ms: float = Field(..., description="Tiempo de invoke del grafo en ms.")
    ok: bool = Field(True, description="False si la petición falló antes de armar la respuesta.")
    error: Optional[str] = Field(None, description="Mensaje de error cuando ok es False.")
