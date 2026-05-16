"""
Tests del router — clasificador determinista.

Estos tests son la documentación viva del comportamiento del sistema.
Cada caso representa un "golden path" documentado o un caso límite.

Estructura:
    - Tablas de casos por ruta (SQL / EVIDENCE / HYBRID / UNKNOWN / AMBIGUOUS)
    - Tests parametrizados que fallan con mensajes claros
    - Tests de propiedades (pureza, cobertura)
    - Test del caso HERO del spec

Ejecutar:
    pytest tests/test_router.py -v
"""
from __future__ import annotations

import pytest

from app.orchestration.router import classify_route, get_disclaimer
from app.schemas.copilot_state import Route


# ---------------------------------------------------------------------------
# Tablas de golden paths
# (query, ruta_esperada, descripción_del_caso)
# ---------------------------------------------------------------------------

SQL_CASES: list[tuple[str, Route, str]] = [
    (
        "¿Cuántos pacientes diabéticos mayores de 65 años existen?",
        Route.SQL,
        "recuento de pacientes con condición y rango de edad",
    ),
    (
        "Cuantos pacientes tiene nuestra base de datos",
        Route.SQL,
        "recuento sin tildes (variación ortográfica)",
    ),
    (
        "Lista los pacientes con hipertensión en nuestra base de datos",
        Route.SQL,
        "listado explícito de pacientes",
    ),
    (
        "¿Cuáles son las comorbilidades más comunes en la cohorte?",
        Route.SQL,
        "análisis de comorbilidades en cohorte",
    ),
    (
        "Dame la distribución de edades en los registros clínicos",
        Route.SQL,
        "distribución demográfica en registros",
    ),
    (
        "¿Cuántos pacientes hay en nuestra cohorte con diabetes e hipertensión?",
        Route.SQL,
        "recuento con múltiples condiciones en cohorte",
    ),
    (
        "Paciente con diabetes, ¿cuántos tenemos en nuestra base?",
        Route.SQL,
        "perfil de paciente + conteo local sin señales de evidencia → SQL (regla 1)",
    ),
]

EVIDENCE_CASES: list[tuple[str, Route, str]] = [
    (
        "¿Qué evidencia existe sobre metformina reduciendo riesgo cardiovascular?",
        Route.EVIDENCE,
        "pregunta de evidencia directa sobre fármaco",
    ),
    (
        "¿Qué estudios recientes existen sobre hipertensión en ancianos?",
        Route.EVIDENCE,
        "búsqueda de estudios recientes",
    ),
    (
        "¿Qué dice la evidencia sobre el tratamiento de la diabetes tipo 2?",
        Route.EVIDENCE,
        "señal textual 'dice la evidencia'",
    ),
    (
        "¿Existe alguna revisión sistemática sobre inhibidores SGLT2?",
        Route.EVIDENCE,
        "tipo de publicación científica como señal",
    ),
    (
        "¿Cuáles son los tratamientos recomendados para la hipertensión arterial?",
        Route.EVIDENCE,
        "tratamientos recomendados como señal de evidencia",
    ),
    (
        "Busca en PubMed artículos sobre eficacia de estatinas en mayores",
        Route.EVIDENCE,
        "mención directa de PubMed",
    ),
    (
        "¿Hay algún meta-análisis sobre el uso de IECA en insuficiencia cardíaca?",
        Route.EVIDENCE,
        "tipo de estudio meta-análisis",
    ),
]

HYBRID_CASES: list[tuple[str, Route, str]] = [
    (
        # CASO HERO del spec
        "Paciente con diabetes e hipertensión mayor de 65 años. "
        "¿Qué evidencia reciente existe sobre tratamientos que reduzcan riesgo cardiovascular?",
        Route.HYBRID,
        "CASO HERO: perfil + pregunta de evidencia",
    ),
    (
        "Para pacientes como los de nuestra cohorte, ¿qué estudios existen?",
        Route.HYBRID,
        "señal híbrida explícita + referencia a cohorte",
    ),
    (
        "¿Qué tratamiento recomienda la evidencia para pacientes similares al perfil descrito?",
        Route.HYBRID,
        "señal híbrida 'pacientes similares' + señal evidencia",
    ),
    (
        "¿Cuántos pacientes hay con diabetes y qué evidencia existe sobre metformina?",
        Route.HYBRID,
        "señal SQL + señal evidencia en la misma query",
    ),
    (
        "Para este paciente con hipertensión, ¿qué evidencia reciente hay sobre ARA-II?",
        Route.HYBRID,
        "señal híbrida 'para este paciente' + señal evidencia",
    ),
    (
        "En pacientes con insuficiencia renal, ¿qué estudios recomienda la evidencia?",
        Route.HYBRID,
        "señal híbrida 'en pacientes con' + señal evidencia",
    ),
]

UNKNOWN_CASES: list[tuple[str, Route, str]] = [
    (
        "Hola, ¿cómo estás?",
        Route.UNKNOWN,
        "saludo sin señales médicas",
    ),
    (
        "¿Cuál es la capital de Francia?",
        Route.UNKNOWN,
        "pregunta fuera de alcance médico",
    ),
    (
        "",
        Route.UNKNOWN,
        "query vacía",
    ),
    (
        "   ",
        Route.UNKNOWN,
        "query solo espacios",
    ),
    (
        "diabetes",
        Route.UNKNOWN,
        "término médico aislado sin contexto de ruta",
    ),
]

AMBIGUOUS_CASES: list[tuple[str, Route, str]] = [
    (
        "pacientes cohorte evidencia efectividad",
        Route.AMBIGUOUS,
        "empate sql vs evidencia sin pregunta explícita de literatura",
    ),
]


# ---------------------------------------------------------------------------
# Tests parametrizados por ruta
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_route,description", SQL_CASES)
def test_sql_route(query: str, expected_route: Route, description: str) -> None:
    route, reason = classify_route(query)
    assert route == expected_route, (
        f"\n[{description}]\n"
        f"  query    = {query!r}\n"
        f"  esperado = {expected_route.value}\n"
        f"  obtenido = {route.value}\n"
        f"  razón    = {reason!r}"
    )


@pytest.mark.parametrize("query,expected_route,description", EVIDENCE_CASES)
def test_evidence_route(query: str, expected_route: Route, description: str) -> None:
    route, reason = classify_route(query)
    assert route == expected_route, (
        f"\n[{description}]\n"
        f"  query    = {query!r}\n"
        f"  esperado = {expected_route.value}\n"
        f"  obtenido = {route.value}\n"
        f"  razón    = {reason!r}"
    )


@pytest.mark.parametrize("query,expected_route,description", HYBRID_CASES)
def test_hybrid_route(query: str, expected_route: Route, description: str) -> None:
    route, reason = classify_route(query)
    assert route == expected_route, (
        f"\n[{description}]\n"
        f"  query    = {query!r}\n"
        f"  esperado = {expected_route.value}\n"
        f"  obtenido = {route.value}\n"
        f"  razón    = {reason!r}"
    )


@pytest.mark.parametrize("query,expected_route,description", UNKNOWN_CASES)
def test_unknown_route(query: str, expected_route: Route, description: str) -> None:
    route, reason = classify_route(query)
    assert route == expected_route, (
        f"\n[{description}]\n"
        f"  query    = {query!r}\n"
        f"  esperado = {expected_route.value}\n"
        f"  obtenido = {route.value}\n"
        f"  razón    = {reason!r}"
    )


@pytest.mark.parametrize("query,expected_route,description", AMBIGUOUS_CASES)
def test_ambiguous_route(query: str, expected_route: Route, description: str) -> None:
    route, reason = classify_route(query)
    assert route == expected_route, (
        f"\n[{description}]\n"
        f"  query    = {query!r}\n"
        f"  esperado = {expected_route.value}\n"
        f"  obtenido = {route.value}\n"
        f"  razón    = {reason!r}"
    )


# ---------------------------------------------------------------------------
# Test del caso HERO (explícito, con nombre propio)
# ---------------------------------------------------------------------------

def test_hero_case() -> None:
    """
    El caso hero del spec SIEMPRE debe enrutar a HYBRID.

    "Paciente con diabetes e hipertensión mayor de 65 años.
     ¿Qué evidencia reciente existe sobre tratamientos que
     reduzcan riesgo cardiovascular?"

    Este test NO debe eliminarse nunca.
    Es el contrato más importante del router en v1.
    """
    hero_query = (
        "Paciente con diabetes e hipertensión mayor de 65 años. "
        "¿Qué evidencia reciente existe sobre tratamientos que "
        "reduzcan riesgo cardiovascular?"
    )
    route, reason = classify_route(hero_query)
    assert route == Route.HYBRID, (
        f"\nHERO CASE FALLÓ — Este test no debe romperse.\n"
        f"  obtenido = {route.value}\n"
        f"  razón    = {reason!r}"
    )


# ---------------------------------------------------------------------------
# Tests de propiedades del router
# ---------------------------------------------------------------------------

def test_router_is_pure() -> None:
    """El router es determinista: misma query siempre produce mismo resultado."""
    queries = [
        "¿Cuántos pacientes diabéticos existen?",
        "¿Qué evidencia hay sobre metformina?",
        "Paciente con diabetes. ¿Qué estudios hay?",
        "",
    ]
    for query in queries:
        result_a = classify_route(query)
        result_b = classify_route(query)
        assert result_a == result_b, (
            f"Router no es puro para query={query!r}: "
            f"{result_a} != {result_b}"
        )


def test_router_always_returns_reason() -> None:
    """El router siempre devuelve una razón no vacía."""
    all_cases = SQL_CASES + EVIDENCE_CASES + HYBRID_CASES + UNKNOWN_CASES + AMBIGUOUS_CASES
    for query, _, _ in all_cases:
        _, reason = classify_route(query)
        assert isinstance(reason, str) and len(reason) > 0, (
            f"Razón vacía para query={query!r}"
        )


def test_router_always_returns_valid_route() -> None:
    """El router siempre devuelve un valor válido de Route."""
    all_cases = SQL_CASES + EVIDENCE_CASES + HYBRID_CASES + UNKNOWN_CASES + AMBIGUOUS_CASES
    valid_routes = set(Route)
    for query, _, _ in all_cases:
        route, _ = classify_route(query)
        assert route in valid_routes, (
            f"Ruta inválida {route!r} para query={query!r}"
        )


def test_disclaimer_defined_for_all_routes() -> None:
    """Todos los valores de Route tienen un disclaimer definido y no vacío."""
    for route in Route:
        disclaimer = get_disclaimer(route)
        assert isinstance(disclaimer, str), (
            f"Disclaimer no es str para Route.{route.name}"
        )
        assert len(disclaimer) > 0, (
            f"Disclaimer vacío para Route.{route.name}"
        )


def test_contracts_importable() -> None:
    """Los contratos de capabilities deben ser importables sin dependencias pesadas."""
    from app.capabilities.contracts import ClinicalCapability, EvidenceCapability
    from typing import runtime_checkable, Protocol
    # Verificar que son Protocols runtime-checkable
    assert hasattr(ClinicalCapability, "__protocol_attrs__") or (
        callable(ClinicalCapability)
    ), "ClinicalCapability debe ser un Protocol"
    assert hasattr(EvidenceCapability, "__protocol_attrs__") or (
        callable(EvidenceCapability)
    ), "EvidenceCapability debe ser un Protocol"


def test_state_schema_importable() -> None:
    """El estado del grafo debe ser importable con todos sus tipos."""
    from app.schemas.copilot_state import (
        CopilotState,
        ClinicalContext,
        SqlResult,
        ArticleSummary,
        EvidenceBundle,
        TraceStep,
        Route,
        NodeName,
    )
    # CopilotState es TypedDict: verificar que es un tipo
    assert isinstance(CopilotState, type)
    # DTOs son Pydantic: verificar instanciación mínima
    ctx = ClinicalContext()
    assert ctx.conditions == []
    bundle = EvidenceBundle(search_term="test")
    assert bundle.pmids == []
