"""Planificador PubMed heurístico (sin LLM). Usado para vista previa y tests."""
from __future__ import annotations

from typing import Optional, Union

from app.capabilities.evidence_rag.heuristic_evidence_query import preview_pubmed_query
from app.schemas.copilot_state import ClinicalContext


class HeuristicQueryPlanner:
    """Delega en ``preview_pubmed_query`` (stages adaptativos o legacy)."""

    def build_query(
        self,
        free_text: str,
        clinical_context: Optional[Union[ClinicalContext, dict]] = None,
    ) -> str:
        return preview_pubmed_query(free_text, clinical_context)
