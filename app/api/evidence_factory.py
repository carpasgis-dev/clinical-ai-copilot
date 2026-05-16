"""Resuelve la implementación de ``EvidenceCapability`` según configuración."""
from __future__ import annotations

import os

from app.capabilities.contracts import EvidenceCapability
from app.capabilities.evidence_rag.query_planning import resolve_query_planner


def resolve_evidence_backend() -> str:
    """Valor de ``COPILOT_EVIDENCE_BACKEND`` (default: ``ncbi``)."""
    return os.getenv("COPILOT_EVIDENCE_BACKEND", "ncbi").lower().strip()


def build_evidence_capability(backend: str | None = None) -> EvidenceCapability:
    """
    ``stub`` | ``ncbi`` | ``epmc`` | ``multi``.

    ``stub`` para tests/CI sin red; ``ncbi`` (default) para PMIDs reales vía NCBI.

    El planificador de query (heurística / LLM) viene de ``resolve_query_planner()``
    salvo que ``stub`` no lo use.
    """
    b = (backend or resolve_evidence_backend()).lower().strip()
    if b == "stub":
        from app.capabilities.evidence_rag import StubEvidenceCapability

        return StubEvidenceCapability()
    planner = resolve_query_planner()
    if b in ("epmc", "europepmc", "europe_pmc"):
        from app.capabilities.evidence_rag import EuropePmcCapability

        return EuropePmcCapability(planner=planner)
    if b in ("multi", "multisource"):
        from app.capabilities.evidence_rag import MultiSourceEvidenceCapability

        return MultiSourceEvidenceCapability(planner=planner)
    from app.capabilities.evidence_rag import NcbiEvidenceCapability

    return NcbiEvidenceCapability(planner=planner)
