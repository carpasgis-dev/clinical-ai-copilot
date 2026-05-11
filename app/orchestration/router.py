"""
Clasificador de rutas — determinista, sin LLM, sin efectos secundarios.

Contrato público:
    classify_route(query: str) -> tuple[Route, str]

    Devuelve (ruta, razón_legible).
    La razón se escribe directamente en el TraceStep del nodo Router.

Precedencia de reglas (orden importa):
    1. Señales híbridas explícitas        → HYBRID  (mayor prioridad)
    2. Señales SQL *y* evidencia          → HYBRID  (ambas presentes)
    3. Solo señales SQL                   → SQL
    4. Solo señales de evidencia          → EVIDENCE
    5. Sin señales claras                 → UNKNOWN

Filosofía:
    - Función pura: misma entrada → misma salida siempre.
    - Sin imports de BD, PubMed ni LLM.
    - Los falsos positivos en HYBRID son preferibles a los falsos negativos
      (es mejor recuperar evidencia de más que no recuperarla).
    - La tabla de señales es el único lugar para ajustar el comportamiento;
      NO añadir lógica implícita fuera de _SIGNALS y _count_signals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.copilot_state import Route


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
    """

    sql: frozenset[str] = field(default_factory=lambda: frozenset({
        # Conteos y cantidades
        "cuántos", "cuantos", "cuántas", "cuantas",
        # Listados
        "listar", "lista los", "lista las", "listado",
        # Entidades de BD
        "pacientes", "cohorte", "registros",
        # Demografía / estadística de BD
        "demografía", "demografia",
        "distribución", "distribucion",
        "prevalencia en nuestra", "prevalencia de nuestra",
        # Comorbilidades (análisis de BD)
        "comorbilidad", "comorbilidades",
        # Acceso explícito a BD
        "en nuestra base", "en la base de datos", "en nuestra cohorte",
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
        # Frases que mezclan explícitamente perfil de paciente + evidencia
        "paciente con",          # "Paciente con diabetes... ¿qué evidencia?"
        "para este paciente",
        "para mi paciente",
        "pacientes similares",
        "perfil similar",
        "para pacientes como",
        "aplica a pacientes",
        "relevante para pacientes",
        "en pacientes con",      # "evidencia en pacientes con hipertensión"
        "para este perfil",
        "para el perfil",
    }))


_SIGNALS = RouteSignals()

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
}


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Minúsculas + colapsa espacios múltiples.
    Las tildes se preservan intencionalmente para casar con señales acentuadas.
    """
    return re.sub(r"\s+", " ", text.lower().strip())


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
        (<Route.SQL: 'sql'>, 'sql=2, evidence=0, hybrid_explicit=0 → sql only')

        >>> classify_route("¿Qué evidencia existe sobre metformina?")
        (<Route.EVIDENCE: 'evidence'>, 'sql=0, evidence=1, hybrid_explicit=0 → evidence only')

        >>> classify_route("Paciente con diabetes. ¿Qué evidencia existe?")
        (<Route.HYBRID: 'hybrid'>, 'sql=0, evidence=1, hybrid_explicit=1 → explicit hybrid phrase')
    """
    norm = _normalize(query)

    sql_n = _count_signals(norm, _SIGNALS.sql)
    ev_n = _count_signals(norm, _SIGNALS.evidence)
    hybrid_n = _count_signals(norm, _SIGNALS.hybrid_explicit)

    base_reason = f"sql={sql_n}, evidence={ev_n}, hybrid_explicit={hybrid_n}"

    # Regla 1: frase híbrida explícita (mayor prioridad)
    if hybrid_n >= 1:
        return Route.HYBRID, f"{base_reason} → explicit hybrid phrase"

    # Regla 2: señales de ambas categorías
    if sql_n >= 1 and ev_n >= 1:
        return Route.HYBRID, f"{base_reason} → both sql+evidence signals"

    # Regla 3: solo SQL
    if sql_n >= 1:
        return Route.SQL, f"{base_reason} → sql only"

    # Regla 4: solo evidencia
    if ev_n >= 1:
        return Route.EVIDENCE, f"{base_reason} → evidence only"

    # Regla 5: sin señales
    return Route.UNKNOWN, f"{base_reason} → no clear signals"


def get_disclaimer(route: Route) -> str:
    """
    Devuelve el disclaimer estático para una ruta dada.
    Siempre devuelve un string no vacío.
    Usado por el nodo Safety para componer la respuesta final.
    """
    return _DISCLAIMERS[route]
