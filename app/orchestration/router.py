"""
Clasificador de rutas — determinista, sin LLM, sin efectos secundarios.

Contrato público:
    classify_route(query: str) -> tuple[Route, str]

    Devuelve (ruta, razón_legible).
    La razón se escribe directamente en el TraceStep del nodo Router.

Precedencia de reglas (orden importa):
    1. Solo señales SQL (sin evidencia)   → SQL  (cohortes / analytics primero)
    1b. SQL y evidencia empatados sin pregunta explícita de literatura → AMBIGUOUS (fase 3.3)
    2. Señales SQL *y* evidencia (intención clara) → HYBRID
    3. Evidencia + contexto de paciente   → HYBRID  (p. ej. «paciente con …» + evidencia;
       «paciente con» también cuenta como señal SQL ligera para trazas coherentes con cohort_sql)
    4. Frases puente híbridas explícitas  → HYBRID
    5. Solo señales de evidencia          → EVIDENCE
    6. Sin señales claras                 → UNKNOWN

Filosofía:
    - Función pura: misma entrada → misma salida siempre.
    - Sin imports de BD, PubMed ni LLM.
    - Priorizar SQL cuando hay señales de datos locales sin pedir literatura
      (evita sobre-disparar HYBRID en conteos/cohortes).
    - Marcadores de «caso/paciente» solo empujan a HYBRID si también hay
      señales de evidencia (regla 3); el resto de ajustes vive en señales.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.copilot_state import Route


def _explicit_evidence_question(normalized: str) -> bool:
    """True si el texto pide literatura/evidencia de forma inequívoca (no solo términos sueltos)."""
    needles = (
        "que evidencia",
        "qué evidencia",
        "que estudios",
        "qué estudios",
        "evidencia existe",
        "existe evidencia",
        "hay evidencia",
        "que dice la evidencia",
        "qué dice la evidencia",
        "pubmed",
        "literatura",
        "bibliografia",
        "bibliografía",
        "revision sistematica",
        "revisión sistemática",
        "meta-analisis",
        "meta-análisis",
        "guia clinica",
        "guía clínica",
        "recomendacion clinica",
        "recomendación clínica",
        "investigacion reciente",
        "investigación reciente",
    )
    return any(n in normalized for n in needles)


# ---------------------------------------------------------------------------
# Señales de ruta
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteSignals:
    """
    Conjuntos de términos que activan cada ruta.

    Convención:
    - Las señales son substrings a buscar en texto normalizado (minúsculas).
    - Se preservan las tildes: "cuántos" y "cuantos" son señales distintas.
    - Añadir siempre ambas formas si el input del usuario puede variar.
    - Las señales más específicas (frases largas) tienen más peso semántico
      aunque el conteo las trate igual que las cortas.
    - Los marcadores de contexto de paciente para HYBRID viven en
      `_PATIENT_CONTEXT_MARKERS` y solo aplican junto a señales de evidencia.
    """

    sql: frozenset[str] = field(default_factory=lambda: frozenset({
        # Conteos y cantidades
        "cuántos", "cuantos", "cuántas", "cuantas",
        # Listados
        "listar", "lista los", "lista las", "listado",
        # Entidades de BD
        "pacientes", "cohorte", "registros",
        # Descripción de caso / población (señal SQL «ligera»: acota cohorte aunque no diga «cuántos»)
        "paciente con",
        "pacientes con",
        "mayor de",
        "mayores de",
        # Demografía / estadística de BD
        "demografía", "demografia",
        "distribución", "distribucion",
        "prevalencia en nuestra", "prevalencia de nuestra",
        # Comorbilidades (análisis de BD)
        "comorbilidad", "comorbilidades",
        # Acceso explícito a BD
        "en nuestra base", "en la base de datos", "en nuestra cohorte",
        # Cohortes / inventario local (peso mayor que marcos de «paciente con»)
        "tenemos", "cuántos tenemos", "cuantos tenemos",
        "tenemos en nuestra base", "tenemos en nuestra cohorte",
        # Historia clínica (como registro, no como concepto médico)
        "historial clínico", "historial clinico",
        "historia clínica", "historia clinica",
        "expediente",
    }))

    evidence: frozenset[str] = field(default_factory=lambda: frozenset({
        # Término directo
        "evidencia",
        # Tipos de publicación científica
        "estudios", "estudio", "ensayo clínico", "ensayo clinico",
        "meta-análisis", "metaanálisis", "metanalisis",
        "revisión sistemática", "revision sistematica",
        "artículo", "articulo", "publicación", "publicacion",
        # Fuentes
        "pubmed", "literatura", "bibliografía", "bibliografia",
        # Conceptos de EBM
        "eficacia", "efectividad", "efecto", "riesgo relativo",
        "guía clínica", "guia clinica", "recomendación clínica",
        "recomendacion clinica",
        # Frases de consulta de evidencia
        "qué dice la evidencia", "que dice la evidencia",
        "qué estudios", "que estudios",
        "investigación reciente", "investigacion reciente",
        "tratamiento recomendado", "tratamientos recomendados",
    }))

    hybrid_explicit: frozenset[str] = field(default_factory=lambda: frozenset({
        # Frases puente explícitas (literatura / guías en subpoblaciones).
        # No incluir «paciente con» suelta: con evidencia se cubre vía
        # _patient_context_with_evidence; sin evidencia no debe forzar HYBRID.
        "qué evidencia existe para pacientes con",
        "que evidencia existe para pacientes con",
        "qué estudios hay en pacientes con",
        "que estudios hay en pacientes con",
        "qué estudios existen para pacientes con",
        "que estudios existen para pacientes con",
        "tratamiento recomendado para pacientes con",
        "tratamientos recomendados para pacientes con",
        "evidencia para pacientes con",
        "evidencia en pacientes con",
        "pacientes similares",
        "perfil similar",
        "para pacientes como",
        "aplica a pacientes",
        "relevante para pacientes",
        "para este perfil",
        "para el perfil",
    }))


_SIGNALS = RouteSignals()

# Marcadores de caso clínico / subpoblación; solo activan HYBRID si además
# hay señales de evidencia (ev_n >= 1). Así «paciente con … ¿cuántos?»
# no fuerza literatura.
_PATIENT_CONTEXT_MARKERS: frozenset[str] = frozenset({
    "paciente con",
    "para este paciente",
    "para mi paciente",
    "este paciente",
    "mi paciente",
    "en pacientes con",
})


def _patient_context_with_evidence(normalized: str) -> bool:
    """True si el texto ancla un paciente/subpoblación explícita."""
    return any(m in normalized for m in _PATIENT_CONTEXT_MARKERS)


# ---------------------------------------------------------------------------
# Disclaimers estáticos por ruta
# ---------------------------------------------------------------------------

_DISCLAIMERS: dict[Route, str] = {
    Route.SQL: (
        "Los datos mostrados provienen de la base de datos clínica local. "
        "Esta información es orientativa y no constituye consejo médico."
    ),
    Route.EVIDENCE: (
        "Los resultados se basan en búsqueda bibliográfica (PubMed). "
        "Consulte siempre con un profesional sanitario antes de tomar decisiones clínicas."
    ),
    Route.HYBRID: (
        "La respuesta combina datos clínicos estructurados y evidencia bibliográfica (PubMed). "
        "Las citas incluyen PMIDs verificables. "
        "Esta información es orientativa y no constituye consejo médico."
    ),
    Route.UNKNOWN: (
        "No se pudo determinar el tipo de consulta. "
        "Por favor reformule la pregunta o consulte con un profesional sanitario."
    ),
    Route.AMBIGUOUS: (
        "La consulta admite varias interpretaciones (datos locales vs evidencia científica). "
        "Indique su preferencia antes de continuar."
    ),
}


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def normalize_query(text: str) -> str:
    """
    Minúsculas + colapsa espacios múltiples.
    Las tildes se preservan intencionalmente para casar con señales acentuadas.
    """
    return re.sub(r"\s+", " ", text.lower().strip())


_normalize = normalize_query  # alias interno por compatibilidad


def _count_signals(normalized: str, signals: frozenset[str]) -> int:
    """Cuenta cuántas señales del conjunto aparecen en el texto normalizado."""
    return sum(1 for s in signals if s in normalized)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def classify_route(query: str) -> tuple[Route, str]:
    """
    Clasifica la ruta de una consulta de usuario.

    Función pura: sin efectos secundarios, sin I/O, sin LLM.

    Args:
        query: Texto libre del usuario.

    Returns:
        (Route, reason) donde reason es una cadena legible para el TraceStep.

    Examples:
        >>> classify_route("¿Cuántos pacientes diabéticos mayores de 65 años existen?")
        (<Route.SQL: 'sql'>, ...)

        >>> classify_route("¿Qué evidencia existe sobre metformina?")
        (<Route.EVIDENCE: 'evidence'>, ...)

        >>> classify_route("Paciente con diabetes. ¿Qué evidencia existe?")
        (<Route.HYBRID: 'hybrid'>, ...)
    """
    norm = _normalize(query)

    sql_n = _count_signals(norm, _SIGNALS.sql)
    ev_n = _count_signals(norm, _SIGNALS.evidence)
    hybrid_n = _count_signals(norm, _SIGNALS.hybrid_explicit)

    base_reason = f"sql={sql_n}, evidence={ev_n}, hybrid_explicit={hybrid_n}"

    # Regla 1: datos locales / analytics sin pedir literatura
    if sql_n >= 1 and ev_n == 0:
        return Route.SQL, f"{base_reason} → sql only (no evidence signals)"

    # Regla 1b: empate SQL/evidencia sin puente híbrido ni pregunta explícita → aclarar
    if sql_n >= 1 and ev_n >= 1:
        strong_hybrid = hybrid_n >= 1 or _patient_context_with_evidence(norm)
        explicit_lit = _explicit_evidence_question(norm)
        if (
            not strong_hybrid
            and not explicit_lit
            and sql_n == ev_n
        ):
            return (
                Route.AMBIGUOUS,
                f"{base_reason} → ambiguous (tied sql/evidence counts, no explicit literature ask)",
            )

    # Regla 2: señales SQL y de evidencia en la misma consulta
    if sql_n >= 1 and ev_n >= 1:
        return Route.HYBRID, f"{base_reason} → hybrid (sql + evidence signals)"

    # Regla 3: evidencia con marco de paciente / subpoblación
    if ev_n >= 1 and _patient_context_with_evidence(norm):
        return Route.HYBRID, f"{base_reason} → hybrid (evidence + patient context)"

    # Regla 4: frases puente híbrido muy explícitas
    if hybrid_n >= 1:
        return Route.HYBRID, f"{base_reason} → hybrid (explicit bridge phrase)"

    # Regla 5: solo evidencia
    if ev_n >= 1:
        return Route.EVIDENCE, f"{base_reason} → evidence only"

    # Regla 6: sin señales
    return Route.UNKNOWN, f"{base_reason} → no clear signals"


def get_disclaimer(route: Route) -> str:
    """
    Devuelve el disclaimer estático para una ruta dada.
    Siempre devuelve un string no vacío.
    Usado por el nodo Safety para componer la respuesta final.
    """
    return _DISCLAIMERS[route]
