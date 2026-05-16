"""
Contratos de estado del grafo — Clinical Evidence Copilot v1.

CopilotState define exactamente qué información viaja entre nodos.
Los DTOs (Pydantic) garantizan los límites de contexto.
Los nodos SOLO escriben sus propios campos.

Compatibilidad: Python 3.9+, Pydantic v2.

Límites de contexto documentados (anti-prompt-gigante):
    settings.sql_max_rows        = 50   filas máx en SqlResult
    settings.evidence_max_art    = 6    artículos máx en EvidenceBundle (salida final)
    _EVIDENCE_RETRIEVAL_POOL_MAX = 200  candidatos pre-rerank (pool PubMed antes de embed)
    settings.article_max_snippet = 500  chars de abstract por artículo
    settings.clinical_max_list   = 10   ítems por lista en ClinicalContext
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

from app.config.settings import settings

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Route(str, Enum):
    """Ruta que el orquestador elige para cada consulta."""
    SQL = "sql"
    EVIDENCE = "evidence"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


class NodeName(str, Enum):
    """Nombres canónicos de los nodos del grafo LangGraph."""
    ROUTER = "router"
    PLANNER = "planner"
    SQL = "sql"
    CLINICAL_SUMMARY = "clinical_summary"
    PUBMED_QUERY_BUILDER = "pubmed_query_builder"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    SYNTHESIS = "synthesis"
    SAFETY = "safety"
    FALLBACK = "fallback"
    CLARIFY = "clarify"
    REASONING = "reasoning"


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

class ClinicalContext(BaseModel):
    """
    Resumen clínico estructurado que viaja entre nodos.

    Regla: NUNCA contiene filas crudas ni dumps de BD.
    Solo viajan agregados mínimos necesarios para construir la query PubMed.

    Los campos ``population_*`` reflejan la cohorte acotada (p. ej. híbrido con SQL);
    ``conditions`` / ``medications`` siguen siendo hints generales desde BD o stub.
    """
    age_range: Optional[str] = None
    """Rango de edad descriptivo: '65-80', '>65', etc."""

    conditions: List[str] = Field(default_factory=list)
    """Condiciones clínicas detectadas: ['diabetes tipo 2', 'hipertensión']."""

    medications: List[str] = Field(default_factory=list)
    """Medicamentos relevantes: ['metformina', 'enalapril']."""

    population_size: Optional[int] = None
    """Número de pacientes en la cohorte consultada (conteo SQL) cuando aplica."""

    population_hint: Optional[str] = None
    """Descripción breve del perfil para enriquecer la query de PubMed."""

    population_conditions: List[str] = Field(default_factory=list)
    """Condiciones de cohorte para UI/prompts: se condensan prefijos redundantes (p. ej. ``diabet``+``diabetes`` → solo el término más largo); el SQL usa la lista completa del ``CohortQuery``."""

    population_medications: List[str] = Field(default_factory=list)
    """Medicación de cohorte para UI/prompts (misma condensación por prefijo que ``population_conditions`` cuando aplica)."""

    population_age_min: Optional[int] = None
    """Edad mínima inclusiva (años) de la cohorte."""

    population_age_max: Optional[int] = None
    """Edad máxima exclusiva (años) de la cohorte, si se filtró."""

    population_sex: Optional[str] = None
    """``F`` / ``M`` si la cohorte filtró por sexo."""

    @field_validator("conditions", "medications", mode="before")
    @classmethod
    def _cap_list(cls, v: Any) -> List[str]:
        return (v or [])[:settings.clinical_max_list]

    @field_validator("population_conditions", "population_medications", mode="before")
    @classmethod
    def _cap_population_lists(cls, v: Any) -> List[str]:
        return (v or [])[:settings.clinical_max_list]


class SqlResult(BaseModel):
    """
    Resultado de ejecución SQL.
    Las filas están limitadas a settings.sql_max_rows para evitar prompts gigantes.
    """
    executed_query: str
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    tables_used: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    @field_validator("rows", mode="before")
    @classmethod
    def _cap_rows(cls, v: Any) -> List[Dict[str, Any]]:
        return (v or [])[:settings.sql_max_rows]


class ArticleSummary(BaseModel):
    """
    Resumen de un artículo PubMed.
    El snippet de abstract está acotado a settings.article_max_snippet chars.
    """
    pmid: str
    title: str
    abstract_snippet: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    open_access: Optional[bool] = None
    """True si la fuente (p. ej. Europe PMC) indica acceso abierto; NCBI puede dejarlo en None."""

    @field_validator("abstract_snippet", mode="before")
    @classmethod
    def _cap_snippet(cls, v: Any) -> str:
        return (v or "")[:settings.article_max_snippet]


class EvidenceBundle(BaseModel):
    """
    Bundle de evidencia PubMed.
    Máximo settings.evidence_max_art artículos en contexto (v1).
    Siempre incluye PMIDs para trazabilidad/citas.
    """
    search_term: str
    pmids: List[str] = Field(default_factory=list)
    articles: List[ArticleSummary] = Field(default_factory=list)
    chunks_used: int = 0
    oa_pdfs_retrieved: int = 0
    retrieval_debug: Optional[Dict[str, Any]] = Field(
        None,
        description="Observabilidad del retrieval (NCBI: outcome, attempts, errors).",
    )

    @field_validator("articles", mode="before")
    @classmethod
    def _cap_articles(cls, v: Any) -> List[Any]:
        return (v or [])[:settings.evidence_max_art]


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
# Reducers LangGraph (append-only; listas concatenadas con operator.add)
# ---------------------------------------------------------------------------

TraceAppend = Annotated[list[TraceStep], add]
"""Pasos de auditoría: cada nodo devuelve solo los nuevos ítems a anexar."""

WarningsAppend = Annotated[list[str], add]
"""Avisos no fatales (timeouts, degradaciones, límites alcanzados)."""


# ---------------------------------------------------------------------------
# Estado principal del grafo (TypedDict — idiomático para LangGraph)
# ---------------------------------------------------------------------------

class CopilotState(TypedDict, total=False):
    """
    Estado completo del grafo LangGraph.

    Convenciones:
    - total=False → cada nodo solo actualiza sus propios campos.
    - Los DTOs Pydantic garantizan límites; nunca pasar objetos crudos.
    - ``trace`` y ``warnings`` usan reducer ``operator.add`` (append-only).
      Cada nodo devuelve solo los elementos nuevos de la lista.
    - 'disclaimer' siempre se propaga aunque los demás campos estén vacíos.

    Campos obligatorios al iniciar el grafo:
        user_query (str): La consulta del usuario.
        session_id (str): Identificador de sesión para trazabilidad.
    """
    # Input
    user_query: str
    session_id: str

    resolved_user_query: str
    """Texto efectivo NL (p. ej. consulta original + aclaración tras ``Route.AMBIGUOUS``)."""

    needs_clarification: bool
    """True si el turno termina pidiendo aclaración al usuario (fase 3.3)."""

    clarification_question: str
    """Pregunta mostrada al usuario cuando ``needs_clarification`` es True."""

    # Routing
    route: Route
    route_reason: str
    """Qué regla/señal determinó la ruta — para el TraceStep del Router."""

    execution_plan: list[dict[str, str | None]]
    """Pasos previstos por el planner explícito (``kind`` + ``reason``); JSON-serializable."""

    # Outputs de capabilities
    clinical_context: ClinicalContext
    """Producido por ClinicalSummaryNode. Solo resumen estructurado."""

    pubmed_query: str
    pubmed_queries_executed: list[str]
    """Producido por PubMedQueryBuilderNode. Query lista para esearch."""

    sql_result: SqlResult
    """Producido por SQLNode. Incluye la query ejecutada para trazabilidad."""

    evidence_bundle: EvidenceBundle
    """Producido por EvidenceRetrievalNode. Incluye PMIDs para citas."""

    clinical_intent: dict[str, Any]
    """Intención clínica PICO-lite (``ClinicalIntent``) extraída para re-ranking y síntesis."""

    clinical_intent_graph: dict[str, Any]
    """Stage 1: grafo clínico (question_type, PICO, política de evidencia, landmarks esperados)."""

    clinical_evidence_frame: dict[str, Any]
    """Marco semántico canónico (``ClinicalEvidenceFrame``) — SSOT para policy/rerank/claims."""

    synthesis_calibration: dict[str, Any]
    """Calibración epistémica post-retrieval (``SynthesisCalibration.to_dict()``)."""

    reasoning_state: dict[str, Any]
    """Razonamiento clínico explícito (fase 3.7); JSON-serializable (``ReasoningState``)."""

    # Pipeline de síntesis
    synthesis_draft: str
    """Draft interno antes del nodo Safety."""

    medical_answer: dict[str, Any]
    """Respuesta médica estructurada (fase 3.4), JSON-serializable para LangGraph."""

    final_answer: str
    """Respuesta final formateada para el usuario."""

    disclaimer: str
    """Siempre presente. Generado por SafetyNode."""

    # Auditoría / governance
    trace: TraceAppend
    """Pasos de ejecución (reducer append-only)."""

    warnings: WarningsAppend
    """Avisos acumulados (reducer append-only)."""

    # Metadata
    created_at: datetime
