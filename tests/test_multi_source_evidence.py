"""Tests ``MultiSourceEvidenceCapability`` (dedupe, merge, prioridad OA)."""
from __future__ import annotations

from app.capabilities.contracts import EvidenceCapability
from app.capabilities.evidence_rag.multi_source_evidence_capability import (
    MultiSourceEvidenceCapability,
)
from app.schemas.copilot_state import ArticleSummary, EvidenceBundle


class _SrcNcbiFake:
    def build_pubmed_query(self, free_text, clinical_context=None):
        return free_text

    def retrieve_evidence(self, pubmed_query, retmax=6, years_back=5):
        return EvidenceBundle(
            search_term=pubmed_query,
            pmids=["111", "222"],
            articles=[
                ArticleSummary(
                    pmid="111",
                    title="short",
                    abstract_snippet="ncbi abstract",
                    year=2020,
                    open_access=False,
                ),
                ArticleSummary(
                    pmid="222",
                    title="only ncbi",
                    abstract_snippet="b",
                    year=2019,
                    open_access=None,
                ),
            ],
        )

    def health_check(self) -> bool:
        return True


class _SrcEpmcFake:
    def build_pubmed_query(self, free_text, clinical_context=None):
        return free_text

    def retrieve_evidence(self, pubmed_query, retmax=6, years_back=5):
        return EvidenceBundle(
            search_term=pubmed_query,
            pmids=["111"],
            articles=[
                ArticleSummary(
                    pmid="111",
                    title="longer title from epmc",
                    abstract_snippet="ncbi abstract merged with epmc extras",
                    year=2021,
                    open_access=True,
                ),
            ],
        )

    def health_check(self) -> bool:
        return True


def test_multi_source_is_protocol() -> None:
    m = MultiSourceEvidenceCapability(sources=[_SrcNcbiFake(), _SrcEpmcFake()])
    assert isinstance(m, EvidenceCapability)


def test_multi_source_dedupe_merge_and_oa_priority() -> None:
    m = MultiSourceEvidenceCapability(sources=[_SrcNcbiFake(), _SrcEpmcFake()])
    b = m.retrieve_evidence("metformin", retmax=6, years_back=0)

    assert len(b.articles) == 2
    by_pmid = {a.pmid: a for a in b.articles}
    merged = by_pmid["111"]
    assert merged.open_access is True
    assert "epmc" in merged.abstract_snippet
    assert "longer" in merged.title.lower()
    assert merged.year == 2021
    assert b.pmids[0] == "111"
    assert b.pmids[1] == "222"


def test_multi_source_sort_oa_before_non_oa_same_year_logic() -> None:
    class A:
        def build_pubmed_query(self, *a, **k):
            return "q"

        def retrieve_evidence(self, q, retmax=6, years_back=5):
            return EvidenceBundle(
                search_term=q,
                pmids=["2", "1"],
                articles=[
                    ArticleSummary(
                        pmid="2", title="closed", abstract_snippet="x", year=2022, open_access=False
                    ),
                    ArticleSummary(
                        pmid="1", title="open", abstract_snippet="y", year=2022, open_access=True
                    ),
                ],
            )

        def health_check(self) -> bool:
            return True

    m = MultiSourceEvidenceCapability(sources=[A()])
    b = m.retrieve_evidence("q", retmax=6, years_back=0)
    assert b.articles[0].open_access is True
    assert b.articles[0].pmid == "1"
