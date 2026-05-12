"""
Grafo LangGraph del copiloto — router determinista + evidencia inyectable.
"""
from __future__ import annotations

from functools import partial
from typing import Literal, Optional

from langgraph.graph import END, START, StateGraph

from app.capabilities.contracts import ClinicalCapability, EvidenceCapability
from app.capabilities.evidence_rag import NcbiEvidenceCapability
from app.orchestration.nodes import (
    evidence_route_node,
    hybrid_clinical_route_node,
    hybrid_evidence_route_node,
    hybrid_pubmed_route_node,
    router_node,
    safety_node,
    sql_route_node,
    synthesis_stub_node,
    unknown_stub_node,
)
from app.schemas.copilot_state import CopilotState, Route


def _dispatch_after_router(
    state: CopilotState,
) -> Literal["sql", "evidence", "hybrid", "unknown"]:
    route = state["route"]
    if route == Route.SQL:
        return "sql"
    if route == Route.EVIDENCE:
        return "evidence"
    if route == Route.HYBRID:
        return "hybrid"
    return "unknown"


def build_copilot_graph(
    evidence: Optional[EvidenceCapability] = None,
    clinical: Optional[ClinicalCapability] = None,
):
    """
    Construye y compila el grafo principal.

    Args:
        evidence: Implementación de ``EvidenceCapability``.
            Por defecto ``NcbiEvidenceCapability()`` (PubMed vía NCBI).
            Alternativa sin tocar el grafo: ``EuropePmcCapability()`` (Europe PMC REST)
            o ``MultiSourceEvidenceCapability()`` (NCBI + Europe PMC fusionados).
            En tests: ``StubEvidenceCapability``.
        clinical: Datos clínicos (p. ej. ``SqliteClinicalCapability`` con ``CLINICAL_DB_PATH``).
            ``None`` mantiene stubs en la ruta SQL y en el resumen híbrido (CI y demos sin BD).

    Returns:
        Grafo compilado listo para ``invoke`` / ``ainvoke``.
    """
    ev: EvidenceCapability = evidence if evidence is not None else NcbiEvidenceCapability()

    builder = StateGraph(CopilotState)

    builder.add_node("router", router_node)
    builder.add_node("sql_route", partial(sql_route_node, clinical=clinical))
    builder.add_node(
        "evidence_stub",
        partial(evidence_route_node, evidence=ev),
    )
    builder.add_node(
        "hybrid_clinical",
        partial(hybrid_clinical_route_node, clinical=clinical),
    )
    builder.add_node(
        "hybrid_pubmed",
        partial(hybrid_pubmed_route_node, evidence=ev),
    )
    builder.add_node(
        "hybrid_evidence",
        partial(hybrid_evidence_route_node, evidence=ev),
    )
    builder.add_node("unknown_stub", unknown_stub_node)
    builder.add_node("synthesis", synthesis_stub_node)
    builder.add_node("safety", safety_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _dispatch_after_router,
        {
            "sql": "sql_route",
            "evidence": "evidence_stub",
            "hybrid": "hybrid_clinical",
            "unknown": "unknown_stub",
        },
    )

    builder.add_edge("sql_route", "synthesis")
    builder.add_edge("evidence_stub", "synthesis")
    builder.add_edge("hybrid_clinical", "hybrid_pubmed")
    builder.add_edge("hybrid_pubmed", "hybrid_evidence")
    builder.add_edge("hybrid_evidence", "synthesis")
    builder.add_edge("unknown_stub", "safety")

    builder.add_edge("synthesis", "safety")
    builder.add_edge("safety", END)

    return builder.compile()
