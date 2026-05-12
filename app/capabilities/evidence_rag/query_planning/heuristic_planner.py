"""Planificador determinista (baseline / safe mode)."""
from __future__ import annotations

from typing import Optional, Union

from app.capabilities.evidence_rag.heuristic_evidence_query import build_evidence_search_query
from app.schemas.copilot_state import ClinicalContext


class HeuristicQueryPlanner:
    """Delega en ``build_evidence_search_query`` (rápido, estable, testeable)."""

    def build_query(
        self,
        free_text: str,
        clinical_context: Optional[Union[ClinicalContext, dict]] = None,
    ) -> str:
        return build_evidence_search_query(free_text, clinical_context)
