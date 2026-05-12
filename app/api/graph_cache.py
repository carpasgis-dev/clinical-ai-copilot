"""Grafo compilado cacheado por backend de evidencia y BD clínica opcional."""
from __future__ import annotations

from functools import lru_cache

from app.api.clinical_factory import build_sqlite_clinical_if_configured
from app.api.evidence_factory import build_evidence_capability
from app.orchestration.graph import build_copilot_graph


@lru_cache(maxsize=48)
def get_compiled_graph(backend_key: str, query_planner_token: str, clinical_token: str):
    """
    Grafo compilado por evidencia, planner y ruta SQLite clínica (si el fichero existe).

    ``query_planner_token`` / ``clinical_token`` deben coincidir con los helpers del caller.
    """
    ev = build_evidence_capability(backend_key)
    clinical = None
    if clinical_token != "none":
        clinical = build_sqlite_clinical_if_configured(clinical_token)
    return build_copilot_graph(evidence=ev, clinical=clinical)


def clear_graph_cache() -> None:
    """Invalida el cache del grafo (tests o tras cambiar variables de entorno)."""
    get_compiled_graph.cache_clear()
