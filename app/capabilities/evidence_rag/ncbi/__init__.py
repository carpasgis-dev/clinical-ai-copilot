"""
Cliente NCBI (PubMed) — código alineado con PRSN 3.0.

Fuente de referencia en el monorepo:
``cursos_actividades/sina_mcp/prsn3.0/src/prsn30/pubmed/``.
"""
from app.capabilities.evidence_rag.ncbi.eutils import (
    esearch_pubmed,
    fetch_pubmed_xml,
    parse_pubmed_fetch_xml,
    search_and_fetch_abstracts,
)

__all__ = [
    "esearch_pubmed",
    "fetch_pubmed_xml",
    "parse_pubmed_fetch_xml",
    "search_and_fetch_abstracts",
]
