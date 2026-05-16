"""Tests Europe PMC (mock HTTP + protocol)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.capabilities.contracts import EvidenceCapability
from app.capabilities.evidence_rag.europe_pmc import EuropePmcCapability


def test_europe_pmc_satisfies_protocol() -> None:
    assert isinstance(EuropePmcCapability(), EvidenceCapability)


def test_europe_pmc_retrieve_evidence_maps_json() -> None:
    payload = {
        "resultList": {
            "result": [
                {
                    "pmid": "12345678",
                    "title": "Example RCT",
                    "abstractText": "Background and methods.",
                    "pubYear": "2023",
                    "doi": "10.1234/example",
                    "isOpenAccess": "Y",
                }
            ]
        }
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=payload)

    cap = EuropePmcCapability()
    hits = payload["resultList"]["result"]
    with patch(
        "app.capabilities.evidence_rag.retrieval_parallel.parallel_retrieval_enabled",
        return_value=False,
    ), patch(
        "app.capabilities.evidence_rag.europe_pmc.search_europe_pmc",
        return_value=hits,
    ):
        bundle = cap.retrieve_evidence("metformin diabetes", retmax=3, years_back=0)

    assert bundle.search_term == "metformin diabetes"
    assert bundle.pmids == ["12345678"]
    assert len(bundle.articles) == 1
    a = bundle.articles[0]
    assert a.pmid == "12345678"
    assert a.title == "Example RCT"
    assert "Background" in a.abstract_snippet
    assert a.year == 2023
    assert a.doi == "10.1234/example"
    assert a.open_access is True


def test_europe_pmc_retrieve_empty_query() -> None:
    cap = EuropePmcCapability()
    b = cap.retrieve_evidence("   ", retmax=3, years_back=0)
    assert b.pmids == []
    assert b.articles == []
