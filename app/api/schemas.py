"""Esquemas Pydantic para ``POST /query``."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

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
        description=(
            "Consulta en lenguaje natural (obligatoria, mínimo 1 carácter). "
            "Si falta o está vacía, FastAPI responde **422 Unprocessable Entity**."
        ),
        json_schema_extra={"examples": [HERO_QUERY_EXAMPLE]},
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Id de sesión para trazas y logs; si se omite se genera uno.",
    )


class Http422DetailItem(BaseModel):
    """
    Un elemento de ``detail`` en las respuestas **422** de FastAPI
    (``RequestValidationError`` / errores de ``QueryRequest``).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "loc": ["body", "query"],
                    "msg": "Field required",
                    "type": "missing",
                },
                {
                    "loc": ["body", "query"],
                    "msg": "String should have at least 1 character",
                    "type": "string_too_short",
                    "input": "",
                    "ctx": {"min_length": 1},
                    "url": "https://errors.pydantic.dev/2/v/string_too_short",
                },
            ]
        }
    )

    loc: List[Any] = Field(
        ...,
        description='Ruta al fallo, p. ej. ``["body","query"]`` o ``["body",1]`` si el JSON es inválido.',
        examples=[["body", "query"]],
    )
    msg: str = Field(
        ...,
        description="Mensaje legible del error.",
        examples=["Field required"],
    )
    type: str = Field(
        ...,
        description="Código Pydantic (p. ej. ``missing``, ``string_too_short``, ``json_invalid``).",
        examples=["missing"],
    )
    input: Optional[Union[str, int, float, bool, Dict[str, Any], List[Any]]] = Field(
        None,
        description="Valor recibido que no pasó la validación (puede ser cualquier tipo JSON o null).",
        examples=[None],
    )
    ctx: Optional[Dict[str, Any]] = Field(
        None,
        description="Contexto adicional del validador (p. ej. ``{\"min_length\": 1}``), si viene en la respuesta.",
        examples=[None, {"min_length": 1}],
    )
    url: Optional[str] = Field(
        None,
        description="URL de documentación del error en Pydantic, si se incluye en la respuesta.",
        examples=["https://errors.pydantic.dev/2/v/missing"],
    )


class Http422Response(BaseModel):
    """Cuerpo documentado para **422 Unprocessable Entity** (misma forma que devuelve FastAPI)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "detail": [
                        {"loc": ["body", "query"], "msg": "Field required", "type": "missing"}
                    ]
                },
                {
                    "detail": [
                        {
                            "loc": ["body", "query"],
                            "msg": "String should have at least 1 character",
                            "type": "string_too_short",
                            "input": "",
                            "ctx": {"min_length": 1},
                            "url": "https://errors.pydantic.dev/2/v/string_too_short",
                        }
                    ]
                },
            ]
        }
    )

    detail: List[Http422DetailItem] = Field(
        ...,
        description="Uno o más errores (p. ej. ``query`` vacío, campo ausente, JSON mal formado).",
    )


class RootOut(BaseModel):
    """Respuesta de ``GET /`` (metadatos del servicio)."""

    service: str = Field(..., description="Nombre del servicio.")
    docs: str = Field(..., description="Ruta de la documentación Swagger.")
    health: str = Field(..., description="Ruta del endpoint de salud.")
    query: str = Field(..., description="Indicación del endpoint principal (POST /query).")


class HealthOut(BaseModel):
    """Respuesta de ``GET /health`` (solo señales de configuración no sensibles)."""

    status: str = Field(..., description="``ok`` si el proceso responde.")
    copilot_query_planner: str = Field(..., description="Planner de query PubMed (heuristic|llm|…).")
    copilot_llm_profile: str = Field(..., description="Perfil LLM activo.")
    copilot_synthesis: str = Field(
        ...,
        description="Síntesis de respuesta: ``deterministic`` (solo reglas) o ``llm`` (narrativa vía LLM si hay endpoint).",
    )
    copilot_evidence_backend: str = Field(..., description="Backend de evidencia (ncbi|stub|…).")
    llm_base_url_host: str = Field(..., description="Host del ``LLM_BASE_URL`` o ``(unset)``.")
    openai_api_key_set: str = Field(..., description="``true``/``false`` (no expone la clave).")
    clinical_db_loaded: str = Field(..., description="Si hay token de ruta de BD clínica configurado.")
    clinical_db_path: str = Field(..., description="Ruta resuelta o mensaje si no hay BD.")
    clinical_capability_ready: str = Field(
        ...,
        description="``true`` si la capability clínica SQLite está lista para usarse.",
    )


class CitationOut(BaseModel):
    """Cita verificable hacia PubMed."""

    pmid: str
    title: str
    url: str
    year: Optional[int] = None
    doi: Optional[str] = None


class EvidenceStrength(str, Enum):
    """Grado de evidencia (heurística / LLM futuro); opcional en síntesis determinista."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EvidenceStatement(BaseModel):
    """Afirmación vinculada a PMIDs concretos (grounding para auditoría)."""

    statement: str = Field(..., description="Afirmación atribuible a las citas listadas.")
    citation_pmids: List[str] = Field(
        default_factory=list,
        description="PMIDs que respaldan la afirmación (orden de importancia libre).",
    )
    strength: Optional[EvidenceStrength] = Field(
        None,
        description="Fuerza de evidencia si se conoce; None en stub determinista.",
    )


class SqlPreviewRow(BaseModel):
    """
    Fila de muestra SQL en la respuesta HTTP.

    Solo se serializan propiedades declaradas (p. ej. ``cohort_size`` en conteos agregados);
    otras columnas del resultado interno se omiten al construir ``QueryResponse``.
    """

    model_config = ConfigDict(extra="ignore")

    cohort_size: Optional[int] = Field(
        None,
        description="Ejemplo frecuente: tamaño de cohorte devuelto por un COUNT agregado.",
    )


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
    rows: List[SqlPreviewRow] = Field(
        default_factory=list,
        description=f"Hasta {_SQL_PREVIEW_MAX_ROWS} filas de muestra (columnas libres por fila).",
    )


class PlanStepOut(BaseModel):
    """Un paso del plan explícito (fase 3.1) — alineado con el planner del grafo."""

    kind: str = Field(..., description="Identificador del paso (p. ej. cohort_sql, pubmed_query).")
    reason: Optional[str] = Field(
        None, description="Motivo breve del paso, si el planner lo documenta."
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


class MedicalAnswer(BaseModel):
    """
    Respuesta clínica estructurada (fase 3.4) — fuente de verdad frente a ``final_answer``.

    ``final_answer`` debe derivarse vía ``render_medical_answer_to_text`` (legacy legible).
    """

    summary: str = Field(
        ...,
        description="Síntesis breve del turno (cohorte y/o evidencia).",
    )
    cohort_summary: Optional[str] = Field(
        None,
        description="Descripción de la cohorte / filtros SQL locales cuando aplica.",
    )
    evidence_summary: Optional[str] = Field(
        None,
        description="Resumen agregado de hallazgos bibliográficos cuando aplica.",
    )
    cohort_size: Optional[int] = Field(
        None,
        description="Tamaño de cohorte local si el conteo SQL está disponible.",
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Hallazgos clave en viñetas (determinista o LLM futuro).",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recomendaciones orientativas (no prescripción clínica).",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Limitaciones metodológicas y avisos legales.",
    )
    citations: List[CitationOut] = Field(
        default_factory=list,
        description="Metadatos de citas (PubMed).",
    )
    evidence_statements: List[EvidenceStatement] = Field(
        default_factory=list,
        description="Bloques orientativos por PMID (extracto de abstract o título; grounding explícito).",
    )
    uncertainty_notes: List[str] = Field(
        default_factory=list,
        description="Incertidumbre explícita (fase 3.7; reglas deterministas o LLM futuro).",
    )
    applicability_notes: List[str] = Field(
        default_factory=list,
        description="Notas de aplicabilidad cohorte ↔ evidencia (fase 3.7).",
    )


class EvidenceAssessmentOut(BaseModel):
    """Evaluación heurística de un artículo (fase 3.7.a)."""

    pmid: str = Field("", description="PMID evaluado.")
    relevance_score: float = Field(0.0, description="Puntuación de relevancia 0–1 (heurística).")
    study_type: Optional[str] = Field(None, description="Tipo inferido: rct, meta-analysis, cohort, …")
    applicability: Optional[str] = Field(None, description="Nota breve de aplicabilidad a la cohorte.")


class ReasoningStateOut(BaseModel):
    """Estado de razonamiento clínico serializado (determinista, sin LLM en el builder)."""

    cohort_summary: Optional[str] = Field(None, description="Resumen de cohorte / contexto SQL.")
    evidence_assessments: List[EvidenceAssessmentOut] = Field(
        default_factory=list,
        description="Lista de evaluaciones por PMID.",
    )
    uncertainty_notes: List[str] = Field(
        default_factory=list,
        description="Incertidumbres explícitas (evidencia ausente, conflictos, etc.).",
    )
    conflicts: List[str] = Field(default_factory=list, description="Tensiones cohorte ↔ evidencia.")
    applicability_notes: List[str] = Field(
        default_factory=list,
        description="Notas de transferibilidad o limitaciones de aplicación.",
    )
    evidence_quality: Optional[str] = Field(
        None,
        description="Calidad global de la evidencia recuperada (p. ej. mixta, sin_resultados_pubmed).",
    )


class PubMedRetrievalAttemptOut(BaseModel):
    """Un intento de esearch/efetch/parse en la pipeline PubMed."""

    label: str = Field("", description="primary_pdat, fallback_no_pdat, fallback_shorter_boolean, …")
    query: str = Field("", description="Término enviado a ESearch en ese intento.")
    used_pdat: bool = Field(False, description="Si se aplicó filtro de fechas PDAT.")
    mindate: Optional[str] = Field(None, description="Límite inferior PDAT (YYYY/MM/DD).")
    maxdate: Optional[str] = Field(None, description="Límite superior PDAT.")
    result_count: Optional[int] = Field(None, description="Conteo total reportado por NCBI (si aplica).")
    idlist_length: int = Field(0, description="PMIDs devueltos en la página esearch.")
    http_status: Optional[int] = Field(None, description="Código HTTP de esearch.")
    stage_reached: str = Field("", description="esearch | efetch | parse")
    error: Optional[str] = Field(None, description="Mensaje de error del intento, si lo hubo.")
    articles_parsed: int = Field(0, description="Artículos parseados tras efetch en ese intento.")


class PubMedNormalizationOut(BaseModel):
    """Metadatos de la capa PubMed-safe (normalización previa a ESearch)."""

    warnings: List[str] = Field(default_factory=list, description="Avisos (p. ej. paréntesis equilibrados).")
    steps_applied: List[str] = Field(
        default_factory=list,
        description="Pasos ejecutados: sanitize_unicode, tag_unfielded_quoted_phrases, …",
    )


class OperatorCountsOut(BaseModel):
    """
    Conteos de operadores booleanos en ``retrieval_metrics.operator_counts``.

    Las claves JSON son ``and`` y ``or`` (palabras reservadas en Python → alias Pydantic).
    """

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    and_: int = Field(default=0, alias="and")
    or_: int = Field(default=0, alias="or")


class PubMedRetrievalMetricsOut(BaseModel):
    """Métricas heurísticas de la query (sin llamada extra a NCBI)."""

    query_complexity: float = Field(
        0.0,
        description="Complejidad booleana heurística (típicamente 0–1; puede superar 1 en queries muy largas).",
    )
    boolean_depth: int = Field(0, description="Profundidad máxima de paréntesis.")
    estimated_specificity: str = Field("", description="low | medium | high | unknown (heurística).")
    operator_counts: Optional[OperatorCountsOut] = Field(
        None,
        description="Conteo de operadores AND/OR detectados en la query normalizada.",
    )
    last_attempt_result_count: Optional[int] = Field(
        None,
        description="Último ``result_count`` de NCBI en la cadena de intentos.",
    )
    synthesis_pubtype_refine_applied: Optional[bool] = Field(
        None,
        description="True si la segunda búsqueda con tipos Review/Meta-Analysis/RCT sustituyó resultados.",
    )
    synthesis_pubtype_refine_reason: Optional[str] = Field(
        None,
        description="Motivo del refinamiento: high_esearch_hit_count | weak_design_majority_in_page.",
    )
    weak_design_share_primary_page: Optional[float] = Field(
        None,
        description="Fracción de títulos de la primera página clasificados como diseño débil (heurística).",
    )


class SynthesisPubtypeRefineOut(BaseModel):
    """Metadatos del refinamiento opcional por tipo de publicación (PubMed)."""

    model_config = ConfigDict(extra="ignore")

    applied: bool = Field(False, description="Si el segundo intento produjo artículos sustitutivos.")
    reason: Optional[str] = Field(None, description="high_esearch_hit_count | weak_design_majority_in_page")
    refined_query: Optional[str] = Field(None, description="Término enviado al segundo ESearch.")
    kept_primary_after_refine: bool = Field(
        False,
        description="True si el refinamiento falló o no devolvió artículos y se mantuvo el primer lote.",
    )


class PubMedRetrievalStatusOut(BaseModel):
    """Observabilidad del retrieval PubMed (``retrieval_debug`` del bundle)."""

    model_config = ConfigDict(extra="ignore")

    outcome: str = Field(
        "",
        description="success | zero_hits_esearch | http_error | timeout | stub | …",
    )
    attempts: List[PubMedRetrievalAttemptOut] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list, description="Errores acumulados por intento.")
    final_idlist_length: int = Field(0, description="Máximo de PMIDs vistos en esearch.")
    articles_parsed: int = Field(0, description="Artículos finales tras parse.")
    pubmed_query_planned: Optional[str] = Field(None, description="Query del planner antes de normalizar.")
    normalized_query: Optional[str] = Field(None, description="Query normalizada enviada a los intentos.")
    final_query_sent: Optional[str] = Field(None, description="Query del intento exitoso o del último intento.")
    normalization: Optional[PubMedNormalizationOut] = None
    retrieval_metrics: Optional[PubMedRetrievalMetricsOut] = None
    backend: Optional[str] = Field(None, description="stub_evidence u otro backend cuando aplica.")
    synthesis_pubtype_refine: Optional[SynthesisPubtypeRefineOut] = Field(
        None,
        description="Refinamiento condicional con Review/Meta-Analysis/RCT (segundo ESearch).",
    )


class QueryResponse(BaseModel):
    """Respuesta estructurada del copiloto (E2E / UI)."""

    route: str = Field(..., description="Ruta elegida por el router (sql|evidence|hybrid|unknown).")
    route_reason: str = Field(..., description="Explicación breve de por qué se eligió la ruta.")
    execution_plan: List[PlanStepOut] = Field(
        default_factory=list,
        description="Pasos previstos por el planner explícito (trazabilidad / futuro executor).",
    )
    reasoning_state: Optional[ReasoningStateOut] = Field(
        None,
        description="Estado de razonamiento clínico (fase 3.7.a), si se materializó en el grafo.",
    )
    pubmed_query: Optional[str] = Field(
        None,
        description=(
            "Query PubMed canónica usada en retrieval: primera etapa ejecutada del plan "
            "heurístico (no la salida del LLM salvo etapa llm_refine). Coincide con lo "
            "auditado en pubmed_retrieval_status."
        ),
    )
    pubmed_queries_executed: Optional[List[str]] = Field(
        None,
        description=(
            "Todas las queries booleanas ejecutadas en evidence_retrieval (multi-etapa). "
            "pubmed_query es la primera de esta lista."
        ),
    )
    pubmed_url: Optional[str] = Field(
        None,
        description=(
            "URL del buscador web de PubMed con el término usado para el enlace "
            "(en runtime suele coincidir con la query normalizada enviada a ESearch cuando existe "
            "``pubmed_retrieval_status.normalized_query``)."
        ),
    )
    pubmed_retrieval_status: Optional[PubMedRetrievalStatusOut] = Field(
        None,
        description="Observabilidad del retrieval PubMed (outcome, attempts, normalization, métricas).",
    )
    sql_result: Optional[SqlResultOut] = Field(
        None,
        description="SQL ejecutado en ruta `sql` (y muestra de filas); null en evidence/hybrid/unknown.",
    )
    final_answer: str = Field(
        ...,
        description="Texto legible derivado de ``medical_answer`` (render); el disclaimer lo añade Safety.",
    )
    medical_answer: Optional[MedicalAnswer] = Field(
        None,
        description="Fuente de verdad estructurada (cohorte, evidencia, grounding por PMID).",
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
    needs_clarification: bool = Field(
        False,
        description="True si el turno terminó pidiendo aclaración (ruta ambiguous / nodo clarify).",
    )
    clarification_question: Optional[str] = Field(
        None,
        description="Pregunta al usuario cuando needs_clarification es True.",
    )
    session_id: str
    latency_ms: float = Field(..., description="Tiempo de invoke del grafo en ms.")
    ok: bool = Field(True, description="False si la petición falló antes de armar la respuesta.")
    error: Optional[str] = Field(None, description="Mensaje de error cuando ok es False.")
