"""Resuelve ``EvidenceQueryPlanner``: heurística por defecto; LLM solo para refinamiento opcional."""
from __future__ import annotations

import os

from app.capabilities.evidence_rag.query_planning.heuristic_planner import HeuristicQueryPlanner
from app.capabilities.evidence_rag.query_planning.llm_planner import LlmQueryPlanner
from app.capabilities.evidence_rag.query_planning.protocol import EvidenceQueryPlanner


def query_planner_mode() -> str:
    return os.getenv("COPILOT_QUERY_PLANNER", "heuristic").lower().strip()


def pubmed_llm_refine_enabled() -> bool:
    """Si true, ``evidence_retrieval`` puede añadir una etapa extra con query del LLM."""
    return os.getenv("COPILOT_PUBMED_LLM_REFINE", "").lower().strip() in (
        "1",
        "true",
        "yes",
        "on",
    )


def query_planner_cache_token() -> str:
    profile = os.getenv("COPILOT_LLM_PROFILE", "custom").lower().strip()
    mode = "llm_refine" if pubmed_llm_refine_enabled() else "heuristic"
    return f"{mode}|{profile}"


def resolve_query_planner() -> EvidenceQueryPlanner:
    """
    Planner por defecto: heurística (misma lógica que multi-stage retrieval).

  ``LlmQueryPlanner`` solo se usa en ``evidence_retrieval`` si ``COPILOT_PUBMED_LLM_REFINE=1``.
    """
    return HeuristicQueryPlanner()
