"""Fase 3.3 — texto de aclaración y resolución de respuestas cortas del usuario."""
from __future__ import annotations

from typing import Literal

from app.orchestration.router import normalize_query

ClarificationChoice = Literal["sql", "evidence", "hybrid"]

DEFAULT_CLARIFICATION_QUESTION = (
    "¿Quieres datos de tu cohorte o base de datos local, o evidencia científica (PubMed / literatura)? "
    "Responda p. ej. «solo datos de la cohorte», «solo evidencia» o «ambos»."
)


def parse_clarification_reply(text: str) -> ClarificationChoice | None:
    """
    Interpreta la segunda intervención tras ``Route.AMBIGUOUS`` (reglas simples, sin LLM).
    """
    n = normalize_query(text)
    if not n:
        return None

    hybrid_markers = (
        "ambos",
        "los dos",
        "las dos",
        "hibrido",
        "híbrido",
        "hybrid",
        "datos y evidencia",
        "cohorte y evidencia",
        "sql y evidencia",
    )
    if any(m in n for m in hybrid_markers):
        return "hybrid"

    ev_markers = (
        "solo evidencia",
        "evidencia cientifica",
        "evidencia científica",
        "solo pubmed",
        "pubmed",
        "literatura",
        "bibliografia",
        "bibliografía",
        "estudios cientificos",
        "estudios científicos",
        "solo estudios",
    )
    sql_markers = (
        "solo cohorte",
        "solo datos",
        "datos locales",
        "base de datos",
        "nuestra base",
        "solo sql",
        "solo conteo",
        "solo pacientes",
        "conteo de pacientes",
    )

    wants_sql = any(m in n for m in sql_markers)
    wants_ev = any(m in n for m in ev_markers)

    if wants_sql and not wants_ev:
        return "sql"
    if wants_ev and not wants_sql:
        return "evidence"
    return None
