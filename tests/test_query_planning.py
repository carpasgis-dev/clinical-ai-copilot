"""Planificadores de query de evidencia (heurística, composite, post-proceso)."""
from __future__ import annotations

from app.capabilities.evidence_rag.heuristic_evidence_query import build_evidence_search_query
from app.capabilities.evidence_rag.query_planning import (
    CompositeQueryPlanner,
    HeuristicQueryPlanner,
)
from app.capabilities.evidence_rag.query_planning.llm_postprocess import (
    finalize_llm_pubmed_line,
    pubmed_line_from_llm_text,
)
from app.schemas.copilot_state import ClinicalContext


def test_heuristic_planner_matches_direct_heuristic() -> None:
    h = HeuristicQueryPlanner()
    q = "metformina diabetes"
    ctx = ClinicalContext(conditions=["diabetes tipo 2"], medications=["metformina"])
    assert h.build_query(q, ctx) == build_evidence_search_query(q, ctx)


def test_composite_falls_back_when_primary_raises() -> None:
    class _Boom:
        def build_query(self, free_text, clinical_context=None):
            raise RuntimeError("llm unavailable")

    c = CompositeQueryPlanner(_Boom(), HeuristicQueryPlanner())
    assert c.build_query("insulina", None) == "insulina"


def test_composite_falls_back_when_primary_returns_blank() -> None:
    class _Empty:
        def build_query(self, free_text, clinical_context=None):
            return "   "

    c = CompositeQueryPlanner(_Empty(), HeuristicQueryPlanner())
    assert c.build_query("  warfarina  ", None) == "warfarina"


def test_pubmed_line_from_llm_strips_fence() -> None:
    raw = '```\n("type 2 diabetes") AND ("metformin")\n```'
    assert pubmed_line_from_llm_text(raw) == '("type 2 diabetes") AND ("metformin")'


def test_finalize_llm_pubmed_line_coerces_long_word_list() -> None:
    line = "one two three four five six seven eight"
    out = finalize_llm_pubmed_line(line)
    assert " AND " in out
    assert "(" in out
