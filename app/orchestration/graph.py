"""
Grafo LangGraph del copiloto — router determinista + evidencia inyectable.
"""
from __future__ import annotations

from functools import partial
from typing import Optional

from langgraph.graph import END, START, StateGraph

from app.capabilities.contracts import ClinicalCapability, EvidenceCapability
from app.capabilities.evidence_rag import NcbiEvidenceCapability
from app.orchestration.nodes import (
    executor_node,
    planner_node,
    reasoning_node,
    router_node,
    safety_node,
    synthesis_stub_node,
    synthesis_calibration_node,
)
from app.schemas.copilot_state import CopilotState, Route


def _after_reasoning(state: CopilotState) -> str:
    """Rutas con síntesis cohorte/evidencia vs. cierre directo a safety (unknown / ambiguous)."""
    r = state.get("route")
    if isinstance(r, Route):
        rv = r
    else:
        try:
            rv = Route(str(r))
        except ValueError:
            rv = Route.UNKNOWN
    if rv in (Route.SQL, Route.EVIDENCE, Route.HYBRID):
        return "synthesis"
    return "safety"


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
    builder.add_node("planner", planner_node)
    builder.add_node(
        "executor",
        partial(executor_node, clinical=clinical, evidence=ev),
    )
    builder.add_node("synthesis_calibration", synthesis_calibration_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("synthesis", synthesis_stub_node)
    builder.add_node("safety", safety_node)

    builder.add_edge(START, "router")
    builder.add_edge("router", "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "synthesis_calibration")
    builder.add_edge("synthesis_calibration", "reasoning")
    builder.add_conditional_edges(
        "reasoning",
        _after_reasoning,
        {"synthesis": "synthesis", "safety": "safety"},
    )
    builder.add_edge("synthesis", "safety")
    builder.add_edge("safety", END)

    return builder.compile()
