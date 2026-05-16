"""Deduplicación de evidencia pre-síntesis."""
from __future__ import annotations

from app.orchestration.evidence_dedup import (
    deduplicate_evidence_bundle_dict,
    deduplicate_evidence_statements,
    deduplicate_medical_answer_evidence,
    deduplicate_pmids,
    deduplicate_titles,
)
from app.orchestration.medical_answer_builder import build_stub_medical_answer
from app.schemas.copilot_state import Route


def test_deduplicate_pmids_preserves_order() -> None:
    assert deduplicate_pmids(["1", "2", "1", "", "3", "2"]) == ["1", "2", "3"]


def test_deduplicate_titles_by_pmid_and_identical_title() -> None:
    items = [
        {"pmid": "99", "title": "Trial A"},
        {"pmid": "99", "title": "Trial A (dup)"},
        {"pmid": "100", "title": "Trial B"},
        {"title": "Trial B"},  # mismo título sin pmid → segunda ocurrencia
    ]
    out = deduplicate_titles(items)
    assert len(out) == 2
    assert out[0]["pmid"] == "99"
    assert out[1]["pmid"] == "100"


def test_deduplicate_evidence_statements_by_primary_pmid() -> None:
    rows = [
        {"statement": "Primero PMID 1", "citation_pmids": ["1", "2"]},
        {"statement": "Dup PMID 1", "citation_pmids": ["1"]},
        {"statement": "Segundo PMID 3", "citation_pmids": ["3"]},
    ]
    out = deduplicate_evidence_statements(rows)
    assert len(out) == 2
    assert out[0]["citation_pmids"] == ["1", "2"]
    assert out[1]["citation_pmids"] == ["3"]


def test_deduplicate_evidence_bundle_dict() -> None:
    eb = deduplicate_evidence_bundle_dict(
        {
            "pmids": ["10", "10", "11"],
            "articles": [
                {"pmid": "10", "title": "A"},
                {"pmid": "10", "title": "A again"},
                {"pmid": "11", "title": "B"},
            ],
        }
    )
    assert eb["pmids"] == ["10", "11"]
    assert len(eb["articles"]) == 2


def test_build_stub_medical_answer_dedupes_duplicate_pmids() -> None:
    state = {
        "route": Route.EVIDENCE,
        "user_query": "evidencia sobre diabetes",
        "evidence_bundle": {
            "pmids": ["555", "555", "777"],
            "articles": [
                {"pmid": "555", "title": "Study One", "abstract_snippet": "x" * 20},
                {"pmid": "555", "title": "Study One duplicate", "abstract_snippet": "y" * 20},
                {"pmid": "777", "title": "Study Two", "abstract_snippet": "z" * 20},
            ],
        },
    }
    ma = build_stub_medical_answer(state)
    pmids = [c["pmid"] for c in ma["citations"]]
    assert pmids == deduplicate_pmids(pmids)
    assert len(pmids) == 2
    assert len(ma["evidence_statements"]) == 2
    assert all("555" in s["statement"] or "777" in s["statement"] for s in ma["evidence_statements"])


def test_deduplicate_medical_answer_evidence() -> None:
    ma = deduplicate_medical_answer_evidence(
        {
            "citations": [
                {"pmid": "1", "title": "T"},
                {"pmid": "1", "title": "T dup"},
            ],
            "evidence_statements": [
                {"statement": "a", "citation_pmids": ["1"]},
                {"statement": "b", "citation_pmids": ["1"]},
            ],
        }
    )
    assert len(ma["citations"]) == 1
    assert len(ma["evidence_statements"]) == 1
