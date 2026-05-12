"""Orquesta un planner principal con fallback (p. ej. LLM → heurística)."""
from __future__ import annotations

import logging
from typing import Optional, Union

from app.capabilities.evidence_rag.query_planning.protocol import EvidenceQueryPlanner
from app.schemas.copilot_state import ClinicalContext

_log = logging.getLogger(__name__)


class CompositeQueryPlanner:
    """
    Intenta ``primary``; si lanza, devuelve vacío, o solo espacios, usa ``fallback``.
    """

    def __init__(
        self,
        primary: EvidenceQueryPlanner,
        fallback: EvidenceQueryPlanner,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    def build_query(
        self,
        free_text: str,
        clinical_context: Optional[Union[ClinicalContext, dict]] = None,
    ) -> str:
        try:
            q = self._primary.build_query(free_text, clinical_context)
            if (q or "").strip():
                return q.strip()
            _log.warning(
                "CompositeQueryPlanner: el planner primario devolvió vacío; "
                "fallback a heurística (revisa LLM / OPENAI_API_KEY)."
            )
        except Exception as exc:
            _log.warning(
                "CompositeQueryPlanner: el planner primario falló (%s); fallback a heurística.",
                exc,
            )
        return self._fallback.build_query(free_text, clinical_context).strip()
