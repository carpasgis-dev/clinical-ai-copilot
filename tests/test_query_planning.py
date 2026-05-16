"""Planificador LLM de query de evidencia + utilidades de vocabulario."""
from __future__ import annotations

import pytest

from app.capabilities.evidence_rag.heuristic_evidence_query import build_evidence_search_query
from app.capabilities.evidence_rag.query_planning import (
    HeuristicQueryPlanner,
    resolve_query_planner,
)
from app.capabilities.evidence_rag.query_planning.llm_postprocess import (
    finalize_llm_pubmed_line,
    pubmed_line_from_llm_text,
)
from app.schemas.copilot_state import ClinicalContext


def test_resolve_query_planner_returns_heuristic() -> None:
    assert isinstance(resolve_query_planner(), HeuristicQueryPlanner)


def test_build_evidence_query_uses_population_mesh() -> None:
    ctx = ClinicalContext(
        population_conditions=["diabet"],
        population_medications=["metform"],
        population_age_min=65,
        population_sex="F",
        population_size=2,
    )
    q = build_evidence_search_query("riesgo cardiovascular", ctx)
    assert "aged[MeSH Terms]" in q
    assert "female[MeSH Terms]" in q
    assert "diabetes mellitus" in q.lower() or "diabet" in q.lower()
    assert "metformin" in q.lower() or "metform" in q.lower()
    assert "cardiovascular disease" in q.lower() or "heart disease" in q.lower()
    assert "riesgo cardiovascular" not in q.lower()


def test_expand_fibrilacion_auricular_label_not_fibrilacionauricular() -> None:
    from app.capabilities.evidence_rag.mesh_lite import expand_cohort_token_for_pubmed

    q = expand_cohort_token_for_pubmed("fibrilación auricular")
    assert "atrial fibrillation" in q
    assert "fibrilacionauricular" not in q


def test_af_anticoag_query_for_doac_vs_warfarin_question() -> None:
    ctx = ClinicalContext(
        population_conditions=["diabetes", "hipertensión", "fibrilación auricular"],
        population_medications=["Warfarin"],
        population_age_min=75,
        population_size=0,
    )
    q = build_evidence_search_query(
        "¿Qué evidencia existe sobre anticoagulantes orales directos frente a warfarina "
        "en fibrilación auricular para prevención de ictus?",
        ctx,
    )
    assert "atrial fibrillation" in q
    assert "direct oral anticoagulant" in q or "DOAC" in q
    assert "fibrilacionauricular" not in q


def test_structured_population_ignores_global_medication_noise() -> None:
    ctx = ClinicalContext(
        population_conditions=["diabetes", "hipertensión"],
        population_age_min=65,
        population_size=10,
        medications=["0.25 ML Leuprolide Acetate 30 MG/ML Prefilled Syringe"],
        conditions=["unrelated condition label"],
    )
    q = build_evidence_search_query(
        "¿Qué evidencia reciente existe sobre tratamientos que reduzcan riesgo cardiovascular?",
        ctx,
    )
    assert "Leuprolide" not in q
    assert "unrelated" not in q.lower()
    assert (
        "therapy[tiab]" in q
        or "meta-analysis" in q.lower()
        or "systematic review" in q.lower()
    )
    assert q.count(" AND ") >= 1


def test_pubmed_line_from_llm_strips_fence() -> None:
    raw = '```\n("type 2 diabetes") AND ("metformin")\n```'
    assert pubmed_line_from_llm_text(raw) == '("type 2 diabetes") AND ("metformin")'


def test_finalize_llm_pubmed_line_coerces_long_word_list() -> None:
    line = "one two three four five six seven eight"
    out = finalize_llm_pubmed_line(line)
    assert " AND " in out
    assert "(" in out


def test_finalize_llm_pubmed_line_strips_recent_inside_quotes() -> None:
    line = (
        '("recent cardiovascular risk reduction treatments") AND '
        '("cardiovascular disease prevention" OR "heart disease")'
    )
    out = finalize_llm_pubmed_line(line)
    assert "recent" not in out.lower()
    assert "cardiovascular" in out.lower()
