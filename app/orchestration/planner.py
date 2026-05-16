"""
Planner explícito (fase 3.1): traduce la ruta del router en un ``ExecutionPlan`` serializable.

El nodo ``executor`` (fase 3.6) recorre este plan y aplica los pasos en orden; el plan
es la fuente de verdad de la orquestación (trazas y API).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.schemas.copilot_state import Route


@dataclass(frozen=True)
class PlanStep:
    """Un paso previsto de la ejecución (consumido por ``execute_plan``)."""

    kind: str
    reason: str | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    """Secuencia ordenada de pasos ejecutada por ``executor_node`` / ``execute_plan``."""

    steps: tuple[PlanStep, ...]


def build_execution_plan(route: Route) -> ExecutionPlan:
    """
    Construye el plan según la ruta ya clasificada por el router determinista.

    Los ``kind`` son identificadores estables para API/trazas (snake_case).
    """
    if route == Route.SQL:
        return ExecutionPlan(
            steps=(
                PlanStep("cohort_sql", "conteo / cohorte vía SqliteClinicalCapability"),
                PlanStep("synthesis", "borrador de respuesta"),
                PlanStep("safety", "disclaimer y cierre"),
            )
        )
    if route == Route.EVIDENCE:
        return ExecutionPlan(
            steps=(
                PlanStep("pubmed_query", "construcción de query PubMed"),
                PlanStep("evidence_retrieval", "búsqueda y bundle de artículos"),
                PlanStep("synthesis", "borrador de respuesta"),
                PlanStep("safety", "disclaimer y cierre"),
            )
        )
    if route == Route.HYBRID:
        return ExecutionPlan(
            steps=(
                PlanStep("cohort_sql", "cohorte local acotada (NL estructurado → SQL)"),
                PlanStep("clinical_summary", "contexto clínico + población para evidencia"),
                PlanStep("pubmed_query", "query PubMed condicionada por contexto"),
                PlanStep("evidence_retrieval", "recuperación de evidencia"),
                PlanStep("synthesis", "fusión cohorte + evidencia (stub)"),
                PlanStep("safety", "disclaimer y cierre"),
            )
        )
    if route == Route.AMBIGUOUS:
        return ExecutionPlan(
            steps=(
                PlanStep("clarify", "preguntar intención: datos locales vs evidencia"),
                PlanStep("safety", "disclaimer y cierre"),
            )
        )
    # UNKNOWN
    return ExecutionPlan(
        steps=(
            PlanStep("fallback_unknown", "consulta sin señales claras"),
            PlanStep("safety", "disclaimer y cierre"),
        )
    )


def execution_plan_to_jsonable(plan: ExecutionPlan) -> list[dict[str, str | None]]:
    """Lista JSON-serializable para ``CopilotState`` / API."""
    return [{"kind": s.kind, "reason": s.reason} for s in plan.steps]
