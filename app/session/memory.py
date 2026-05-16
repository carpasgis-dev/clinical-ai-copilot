"""
Memoria de sesión (fase 3.2 + 3.5): estado clínico y conversación por ``session_id`` en RAM.

Sin Redis ni BD: diccionario en proceso con lock para la API.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

from app.capabilities.clinical_sql.cohort_parser import CohortQuery
from app.schemas.copilot_state import Route

if TYPE_CHECKING:
    from app.orchestration.planner import ExecutionPlan

_LOCK = Lock()
_MEMORY: dict[str, "SessionMemory"] = {}


def _copy_cohort(c: CohortQuery | None) -> CohortQuery | None:
    if c is None:
        return None
    return CohortQuery(
        condition_like_tokens=list(c.condition_like_tokens),
        medication_like_tokens=list(c.medication_like_tokens),
        age_min_years=c.age_min_years,
        age_max_years=c.age_max_years,
        sex=c.sex,
        count_only=c.count_only,
        alive_only=c.alive_only,
    )


@dataclass
class SessionMemory:
    """
    Estado conversacional por sesión (cohorte activa + último turno materializado).

    ``last_cohort`` equivale a *active_cohort* del roadmap (cohorte NL→SQL vigente).
    """

    last_cohort: CohortQuery | None = None
    last_route: Route | None = None
    last_plan: ExecutionPlan | None = None
    last_sql_executed: str | None = None
    last_query: str | None = None
    last_sql_result: dict[str, Any] | None = None
    last_medical_answer: dict[str, Any] | None = None
    pending_clarification: bool = False
    """True si el último turno pidió aclaración (``Route.AMBIGUOUS``)."""
    pending_ambiguous_query: str | None = None
    """Consulta original que disparó la ambigüedad (fusionar con la respuesta del usuario)."""


def clear_session_memory_store() -> None:
    """Vacía el almacén (tests o reinicio manual)."""
    with _LOCK:
        _MEMORY.clear()


def load_session_memory(session_id: str) -> SessionMemory:
    """Copia defensiva para que los nodos no muten el store al usar ``last_cohort``."""
    sid = (session_id or "").strip()
    if not sid:
        return SessionMemory()
    with _LOCK:
        mem = _MEMORY.get(sid)
    if mem is None:
        return SessionMemory()
    return SessionMemory(
        last_cohort=_copy_cohort(mem.last_cohort),
        last_route=mem.last_route,
        last_plan=mem.last_plan,
        last_sql_executed=mem.last_sql_executed,
        last_query=mem.last_query,
        last_sql_result=deepcopy(mem.last_sql_result) if mem.last_sql_result else None,
        last_medical_answer=deepcopy(mem.last_medical_answer) if mem.last_medical_answer else None,
        pending_clarification=mem.pending_clarification,
        pending_ambiguous_query=mem.pending_ambiguous_query,
    )


def update_session_after_planner(
    session_id: str,
    route: Route,
    plan: "ExecutionPlan",
) -> None:
    """Tras planificar: conserva cohorte/SQL previos, actualiza ruta y plan."""
    sid = (session_id or "").strip()
    if not sid:
        return
    with _LOCK:
        cur = _MEMORY.get(sid, SessionMemory())
        _MEMORY[sid] = SessionMemory(
            last_cohort=_copy_cohort(cur.last_cohort),
            last_route=route,
            last_plan=plan,
            last_sql_executed=cur.last_sql_executed,
            last_query=cur.last_query,
            last_sql_result=deepcopy(cur.last_sql_result) if cur.last_sql_result else None,
            last_medical_answer=deepcopy(cur.last_medical_answer) if cur.last_medical_answer else None,
            pending_clarification=cur.pending_clarification,
            pending_ambiguous_query=cur.pending_ambiguous_query,
        )


def update_session_after_sql_route(
    session_id: str,
    *,
    route: Route,
    effective_cohort: CohortQuery,
    sql_executed: str | None,
    structured_cohort_applied: bool,
) -> None:
    """
    Tras ejecutar SQL en ruta ``sql`` o fase clínica del híbrido.

    ``structured_cohort_applied``: True si el conteo vino del NL estructurado
    (filtros de cohorte), no del conteo global de vivos.
    """
    sid = (session_id or "").strip()
    if not sid:
        return
    with _LOCK:
        cur = _MEMORY.get(sid, SessionMemory())
        new_cohort = (
            _copy_cohort(effective_cohort)
            if structured_cohort_applied
            else _copy_cohort(cur.last_cohort)
        )
        sql_line = (sql_executed or "").strip() or cur.last_sql_executed
        _MEMORY[sid] = SessionMemory(
            last_cohort=new_cohort,
            last_route=route,
            last_plan=cur.last_plan,
            last_sql_executed=sql_line,
            last_query=cur.last_query,
            last_sql_result=deepcopy(cur.last_sql_result) if cur.last_sql_result else None,
            last_medical_answer=deepcopy(cur.last_medical_answer) if cur.last_medical_answer else None,
            pending_clarification=cur.pending_clarification,
            pending_ambiguous_query=cur.pending_ambiguous_query,
        )


def set_pending_clarification(session_id: str, ambiguous_query: str) -> None:
    """Marca sesión en espera de aclaración (consulta que fue ``Route.AMBIGUOUS``)."""
    sid = (session_id or "").strip()
    if not sid:
        return
    q = (ambiguous_query or "").strip()
    if not q:
        return
    with _LOCK:
        cur = _MEMORY.get(sid, SessionMemory())
        _MEMORY[sid] = SessionMemory(
            last_cohort=_copy_cohort(cur.last_cohort),
            last_route=cur.last_route,
            last_plan=cur.last_plan,
            last_sql_executed=cur.last_sql_executed,
            last_query=cur.last_query,
            last_sql_result=deepcopy(cur.last_sql_result) if cur.last_sql_result else None,
            last_medical_answer=deepcopy(cur.last_medical_answer) if cur.last_medical_answer else None,
            pending_clarification=True,
            pending_ambiguous_query=q,
        )


def clear_pending_clarification(session_id: str) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    with _LOCK:
        cur = _MEMORY.get(sid)
        if cur is None:
            return
        _MEMORY[sid] = SessionMemory(
            last_cohort=_copy_cohort(cur.last_cohort),
            last_route=cur.last_route,
            last_plan=cur.last_plan,
            last_sql_executed=cur.last_sql_executed,
            last_query=cur.last_query,
            last_sql_result=deepcopy(cur.last_sql_result) if cur.last_sql_result else None,
            last_medical_answer=deepcopy(cur.last_medical_answer) if cur.last_medical_answer else None,
            pending_clarification=False,
            pending_ambiguous_query=None,
        )


def _slim_sql_result(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw or not isinstance(raw, dict):
        return None
    rows = raw.get("rows")
    preview: list[Any] = []
    if isinstance(rows, list):
        preview = rows[:3]
    return {
        "executed_query": str(raw.get("executed_query") or ""),
        "row_count": raw.get("row_count"),
        "tables_used": list(raw.get("tables_used") or []),
        "error": raw.get("error"),
        "rows_preview": preview,
    }


def save_session_turn_snapshot(
    session_id: str,
    *,
    user_query: str,
    sql_result: dict[str, Any] | None,
    medical_answer: dict[str, Any] | None,
) -> None:
    """
    Persiste metadatos del turno tras Safety (consulta, SQL resumido, ``MedicalAnswer``).

    La cohorte activa y el plan siguen actualizándose en nodos anteriores (SQL / planner).
    """
    sid = (session_id or "").strip()
    if not sid:
        return
    q = (user_query or "").strip()
    with _LOCK:
        cur = _MEMORY.get(sid, SessionMemory())
        slim = _slim_sql_result(sql_result)
        ma_copy = deepcopy(medical_answer) if medical_answer else None
        _MEMORY[sid] = SessionMemory(
            last_cohort=_copy_cohort(cur.last_cohort),
            last_route=cur.last_route,
            last_plan=cur.last_plan,
            last_sql_executed=cur.last_sql_executed,
            last_query=q or cur.last_query,
            last_sql_result=slim if slim is not None else cur.last_sql_result,
            last_medical_answer=ma_copy if ma_copy is not None else cur.last_medical_answer,
            pending_clarification=cur.pending_clarification,
            pending_ambiguous_query=cur.pending_ambiguous_query,
        )
