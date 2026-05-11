"""
Contratos de estado del grafo — Clinical Evidence Copilot v1.

CopilotState define exactamente qué información viaja entre nodos.
Los DTOs (Pydantic) garantizan los límites de contexto.
Los nodos SOLO escriben sus propios campos.

Compatibilidad: Python 3.9+, Pydantic v1.

Límites de contexto documentados (anti-prompt-gigante):
    _SQL_MAX_ROWS        = 50   filas máx en SqlResult
    _EVIDENCE_MAX_ART    = 6    artículos máx en EvidenceBundle
    _ARTICLE_MAX_SNIPPET = 500  chars de abstract por artículo
    _CLINICAL_MAX_LIST   = 10   ítems por lista en ClinicalContext
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, validator

# ---------------------------------------------------------------------------
# Límites de contexto — cambiar aquí afecta a todo el sistema
# ---------------------------------------------------------------------------
_SQL_MAX_ROWS: int = 50
_EVIDENCE_MAX_ART: int = 6
_ARTICLE_MAX_SNIPPET: int = 500
_CLINICAL_MAX_LIST: int = 10


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Route(str, Enum):
    """Ruta que el orquestador elige para cada consulta."""
    SQL = "sql"
    EVIDENCE = "evidence"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class NodeName(str, Enum):
    """Nombres canónicos de los nodos del grafo LangGraph."""
    ROUTER = "router"
    CLINICAL_SUMMARY = "clinical_summary"
    PUBMED_QUERY_BUILDER = "pubmed_query_builder"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    SYNTHESIS = "synthesis"
    SAFETY = "safety"


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

class ClinicalContext(BaseModel):
    """
    Resumen clínico estructurado que viaja entre nodos.

    Regla: NUNCA contiene filas crudas ni dumps de BD.
    Solo viajan agregados mínimos necesarios para construir la query PubMed.
    """
    age_range: Optional[str] = None
    """Rango de edad descriptivo: '65-80', '>65', etc."""

    conditions: List[str] = Field(default_factory=list)
    """Condiciones clínicas detectadas: ['diabetes tipo 2', 'hipertensión']."""

    medications: List[str] = Field(default_factory=list)
    """Medicamentos relevantes: ['metformina', 'enalapril']."""

    population_size: Optional[int] = None
    """Solo para consultas poblacionales: número de pacientes encontrados."""

    population_hint: Optional[str] = None
    """Descripción breve del perfil para enriquecer la query de PubMed."""

    @validator("conditions", "medications", pre=True, always=True)
    def _cap_list(cls, v: list) -> list:
        return (v or [])[:_CLINICAL_MAX_LIST]


class SqlResult(BaseModel):
    """
    Resultado de ejecución SQL.
    Las filas están limitadas a _SQL_MAX_ROWS para evitar prompts gigantes.
    """
    executed_query: str
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    tables_used: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    @validator("rows", pre=True, always=True)
    def _cap_rows(cls, v: list) -> list:
        return (v or [])[:_SQL_MAX_ROWS]


class ArticleSummary(BaseModel):
    """
    Resumen de un artículo PubMed.
    El snippet de abstract está acotado a _ARTICLE_MAX_SNIPPET chars.
    """
    pmid: str
    title: str
    abstract_snippet: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None

    @validator("abstract_snippet", pre=True, always=True)
    def _cap_snippet(cls, v: str) -> str:
        return (v or "")[:_ARTICLE_MAX_SNIPPET]


class EvidenceBundle(BaseModel):
    """
    Bundle de evidencia PubMed.
    Máximo _EVIDENCE_MAX_ART artículos en contexto (v1).
    Siempre incluye PMIDs para trazabilidad/citas.
    """
    search_term: str
    pmids: List[str] = Field(default_factory=list)
    articles: List[ArticleSummary] = Field(default_factory=list)
    chunks_used: int = 0
    oa_pdfs_retrieved: int = 0

    @validator("articles", pre=True, always=True)
    def _cap_articles(cls, v: list) -> list:
        return (v or [])[:_EVIDENCE_MAX_ART]


class TraceStep(BaseModel):
    """
    Un paso de auditoría del sistema.

    El trace completo es el audit log del copiloto:
    qué hizo, cuándo, qué fuentes usó, qué SQL ejecutó.
    Esto es lo que hace el sistema explicable y apto para healthcare governance.
    """
    node: NodeName
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_ms: Optional[float] = None
    summary: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Estado principal del grafo (TypedDict — idiomático para LangGraph)
# ---------------------------------------------------------------------------

class CopilotState(TypedDict, total=False):
    """
    Estado completo del grafo LangGraph.

    Convenciones:
    - total=False → cada nodo solo actualiza sus propios campos.
    - Los DTOs Pydantic garantizan límites; nunca pasar objetos crudos.
    - 'trace' es append-only; en v2 se puede añadir un reducer LangGraph.
    - 'disclaimer' siempre se propaga aunque los demás campos estén vacíos.

    Campos obligatorios al iniciar el grafo:
        user_query (str): La consulta del usuario.
        session_id (str): Identificador de sesión para trazabilidad.
    """
    # Input
    user_query: str
    session_id: str

    # Routing
    route: Route
    route_reason: str
    """Qué regla/señal determinó la ruta — para el TraceStep del Router."""

    # Outputs de capabilities
    clinical_context: ClinicalContext
    """Producido por ClinicalSummaryNode. Solo resumen estructurado."""

    sql_result: SqlResult
    """Producido por SQLNode. Incluye la query ejecutada para trazabilidad."""

    evidence_bundle: EvidenceBundle
    """Producido por EvidenceRetrievalNode. Incluye PMIDs para citas."""

    # Pipeline de síntesis
    synthesis_draft: str
    """Draft interno antes del nodo Safety."""

    final_answer: str
    """Respuesta final formateada para el usuario."""

    disclaimer: str
    """Siempre presente. Generado por SafetyNode."""

    # Auditoría / governance
    trace: List[TraceStep]
    """Lista append-only de pasos de ejecución."""

    # Metadata
    created_at: datetime
