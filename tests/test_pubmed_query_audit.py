"""Auditoría: pubmed_query canónica alineada con retrieval ejecutado."""
from __future__ import annotations

from app.capabilities.evidence_rag.heuristic_evidence_query import (
    build_evidence_retrieval_stages,
    canonical_pubmed_query_after_retrieval,
    preview_pubmed_query,
)
from app.schemas.copilot_state import ClinicalContext

_HERO = (
    "En pacientes con diabetes e hipertensión ≥65 años, "
    "¿qué evidencia hay sobre inhibidores SGLT2 o agonistas GLP-1 "
    "frente a solo metformina para reducir eventos cardiovasculares?"
)


def test_canonical_pubmed_query_single_stage() -> None:
    q = '("diabetes"[tiab]) AND ("sglt2"[tiab])'
    canonical, executed = canonical_pubmed_query_after_retrieval([q])
    assert canonical == q
    assert executed == [q]


def test_canonical_pubmed_query_multi_stage_uses_first_executed() -> None:
    q1 = '("strict"[tiab])'
    q2 = '("landmark"[tiab])'
    canonical, executed = canonical_pubmed_query_after_retrieval([q1, q2])
    assert canonical == q1
    assert executed == [q1, q2]


def test_preview_matches_first_adaptive_stage() -> None:
    ctx = ClinicalContext(
        population_conditions=["diabet", "hipertens"],
        population_age_min=65,
    )
    preview = preview_pubmed_query(_HERO, ctx)
    stages = build_evidence_retrieval_stages(_HERO, ctx)
    assert stages
    assert preview == stages[0].query
    assert "empagliflozin" in stages[-1].query.lower() or "EMPA-REG" in stages[-1].query
