"""Contrato del planificador de query (separado de la recuperación)."""
from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

from app.schemas.copilot_state import ClinicalContext


@runtime_checkable
class EvidenceQueryPlanner(Protocol):
    """
    Traduce intención clínica + contexto a una cadena lista para esearch / Europe PMC.

    La recuperación (HTTP, XML, JSON) vive en ``EvidenceCapability.retrieve_evidence``.
    """

    def build_query(
        self,
        free_text: str,
        clinical_context: Optional[Union[ClinicalContext, dict]] = None,
    ) -> str:
        """Devuelve cadena vacía si no hay nada usable."""
        ...
