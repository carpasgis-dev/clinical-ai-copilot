"""Planner explícito (fase 3.1): ``ExecutionPlan`` por ruta."""
from __future__ import annotations

from app.orchestration.planner import (
    ExecutionPlan,
    PlanStep,
    build_execution_plan,
    execution_plan_to_jsonable,
)
from app.schemas.copilot_state import Route


def test_build_execution_plan_sql() -> None:
    p = build_execution_plan(Route.SQL)
    assert [s.kind for s in p.steps] == ["cohort_sql", "synthesis", "safety"]


def test_build_execution_plan_evidence() -> None:
    p = build_execution_plan(Route.EVIDENCE)
    assert [s.kind for s in p.steps] == [
        "pubmed_query",
        "evidence_retrieval",
        "synthesis",
        "safety",
    ]


def test_build_execution_plan_hybrid_matches_user_example_shape() -> None:
    """Alineado con el ejemplo de roadmap (sin nodo safety en el ejemplo del usuario)."""
    p = build_execution_plan(Route.HYBRID)
    kinds = [s.kind for s in p.steps]
    assert kinds[:5] == [
        "cohort_sql",
        "clinical_summary",
        "pubmed_query",
        "evidence_retrieval",
        "synthesis",
    ]
    assert kinds[-1] == "safety"


def test_build_execution_plan_unknown() -> None:
    p = build_execution_plan(Route.UNKNOWN)
    assert [s.kind for s in p.steps] == ["fallback_unknown", "safety"]


def test_build_execution_plan_ambiguous() -> None:
    p = build_execution_plan(Route.AMBIGUOUS)
    assert [s.kind for s in p.steps] == ["clarify", "safety"]


def test_execution_plan_to_jsonable_roundtrip_keys() -> None:
    plan = ExecutionPlan(steps=(PlanStep("x", "y"), PlanStep("z", None)))
    j = execution_plan_to_jsonable(plan)
    assert j == [{"kind": "x", "reason": "y"}, {"kind": "z", "reason": None}]
