from app.capabilities.evidence_rag.ncbi.pubmed_urls import pubmed_web_search_url


def test_pubmed_web_search_url_encodes_term() -> None:
    u = pubmed_web_search_url("diabetes AND heart")
    assert u.startswith("https://pubmed.ncbi.nlm.nih.gov/?term=")
    assert "diabetes" in u
    assert " " not in u.split("term=", 1)[-1] or "%20" in u


def test_pubmed_web_search_url_empty() -> None:
    assert pubmed_web_search_url("") == "https://pubmed.ncbi.nlm.nih.gov/"
