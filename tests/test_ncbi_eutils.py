"""Tests offline del parser XML de PubMed (eutils)."""
from __future__ import annotations

from app.capabilities.evidence_rag.ncbi.eutils import parse_pubmed_fetch_xml


def test_parse_pubmed_fetch_xml_minimal() -> None:
    xml = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">99999999</PMID>
      <Article PubModel="Print-Electronic">
        <ArticleTitle>Test title for parser</ArticleTitle>
        <Abstract>
          <AbstractText Label="RESULTS">Hello world abstract.</AbstractText>
        </Abstract>
        <Journal>
          <JournalIssue>
            <PubDate><Year>2021</Year></PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""
    rows = parse_pubmed_fetch_xml(xml)
    assert len(rows) == 1
    r = rows[0]
    assert r.pmid == "99999999"
    assert "Test title" in r.title
    assert "Hello world" in r.abstract
    assert r.year == "2021"


def test_ncbi_evidence_capability_build_query() -> None:
    from app.capabilities.evidence_rag.ncbi_evidence_capability import NcbiEvidenceCapability
    from app.capabilities.evidence_rag.query_planning import HeuristicQueryPlanner
    from app.schemas.copilot_state import ClinicalContext

    cap = NcbiEvidenceCapability(planner=HeuristicQueryPlanner())
    q = cap.build_pubmed_query(
        "metformina riesgo cardiovascular",
        ClinicalContext(conditions=["diabetes tipo 2"], medications=["metformina"]),
    )
    assert "metformin" in q.lower()
    assert "cardiovascular" in q.lower() or "mace" in q.lower()


def test_ncbi_evidence_capability_retrieve_evidence_empty_query() -> None:
    from app.capabilities.evidence_rag.ncbi_evidence_capability import NcbiEvidenceCapability

    cap = NcbiEvidenceCapability()
    b = cap.retrieve_evidence("   ", retmax=3, years_back=0)
    assert b.search_term == ""
    assert b.pmids == []
    assert b.articles == []
    assert b.retrieval_debug and b.retrieval_debug.get("outcome") == "no_query"


def test_evidence_capability_protocol_instances() -> None:
    from app.capabilities.contracts import EvidenceCapability
    from app.capabilities.evidence_rag import (
        EuropePmcCapability,
        MultiSourceEvidenceCapability,
        NcbiEvidenceCapability,
        StubEvidenceCapability,
    )

    assert isinstance(NcbiEvidenceCapability(), EvidenceCapability)
    assert isinstance(StubEvidenceCapability(), EvidenceCapability)
    assert isinstance(EuropePmcCapability(), EvidenceCapability)
    assert isinstance(MultiSourceEvidenceCapability(), EvidenceCapability)


def test_append_synthesis_pub_types_to_pubmed_query() -> None:
    from app.capabilities.evidence_rag.ncbi.eutils import append_synthesis_pub_types_to_pubmed_query

    q = append_synthesis_pub_types_to_pubmed_query(
        'diabetes[tiab] AND ("blood pressure"[tiab])'
    )
    assert q.startswith("(")
    assert "[tiab]" in q
    assert "Review[pt]" in q and "Meta-Analysis[pt]" in q
    assert " AND " in q


def test_append_synthesis_pub_types_empty() -> None:
    from app.capabilities.evidence_rag.ncbi.eutils import append_synthesis_pub_types_to_pubmed_query

    assert append_synthesis_pub_types_to_pubmed_query("") == ""
    assert append_synthesis_pub_types_to_pubmed_query("   ") == ""
