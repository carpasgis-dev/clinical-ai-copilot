"""Agregación cross-paper por clase terapéutica."""
from __future__ import annotations

from app.capabilities.evidence_rag.evidence_aggregation import aggregate_therapeutic_findings

_HERO_ARTS = [
    {
        "pmid": "1",
        "title": "EMPA-REG OUTCOME: cardiovascular outcomes with empagliflozin",
        "abstract_snippet": "Major adverse cardiovascular events and cardiovascular death in type 2 diabetes.",
    },
    {
        "pmid": "2",
        "title": "DECLARE-TIMI 58: dapagliflozin and MACE",
        "abstract_snippet": "Randomized trial; heart failure hospitalization and cardiovascular death.",
    },
    {
        "pmid": "3",
        "title": "LEADER trial: liraglutide and cardiovascular events",
        "abstract_snippet": "GLP-1 receptor agonist; major adverse cardiovascular events.",
    },
    {
        "pmid": "99",
        "title": "CVOT Summit Report 2024: state of the science",
        "abstract_snippet": "Expert panel report on cardiovascular outcome trials landscape.",
    },
]


def test_aggregates_sglt2_and_glp1_excludes_summit_report() -> None:
    block = aggregate_therapeutic_findings(
        _HERO_ARTS,
        question_type="treatment_efficacy",
    )
    assert block is not None
    assert "SGLT2" in block
    assert "GLP-1" in block or "incretin" in block.lower()
    assert "Summit" not in block
    assert "EMPA-REG" in block or "DECLARE" in block


def test_no_aggregate_for_mechanism_question() -> None:
    assert aggregate_therapeutic_findings(_HERO_ARTS, question_type="mechanism") is None
