"""Resuelve ``EvidenceQueryPlanner`` según ``COPILOT_QUERY_PLANNER``."""
from __future__ import annotations

import os

from app.capabilities.evidence_rag.query_planning.composite_planner import CompositeQueryPlanner
from app.capabilities.evidence_rag.query_planning.heuristic_planner import HeuristicQueryPlanner
from app.capabilities.evidence_rag.query_planning.llm_planner import LlmQueryPlanner
from app.capabilities.evidence_rag.query_planning.protocol import EvidenceQueryPlanner


def query_planner_cache_token() -> str:
    """Token estable para particionar caches (grafo) cuando cambia el planner."""
    return os.getenv("COPILOT_QUERY_PLANNER", "heuristic").lower().strip()


def resolve_query_planner() -> EvidenceQueryPlanner:
    """
    ``COPILOT_QUERY_PLANNER``:

    - ``heuristic`` (default): solo heurística determinista.
    - ``llm``: ``LlmQueryPlanner`` con fallback a heurística.
    - ``llm_only``: solo LLM (sin fallback; fallos propagan excepción).
    """
    mode = query_planner_cache_token()
    h = HeuristicQueryPlanner()
    if mode in ("llm", "llm_composite", "composite"):
        return CompositeQueryPlanner(LlmQueryPlanner(), h)
    if mode in ("llm_only", "llm-only"):
        return LlmQueryPlanner()
    return h
