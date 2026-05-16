"""URLs de depuración / demo para búsquedas PubMed (web)."""
from __future__ import annotations

from urllib.parse import quote


def pubmed_web_search_url(term: str) -> str:
    """URL del buscador web de PubMed con el término codificado (misma semántica aproximada que esearch)."""
    q = (term or "").strip()
    if not q:
        return "https://pubmed.ncbi.nlm.nih.gov/"
    return "https://pubmed.ncbi.nlm.nih.gov/?term=" + quote(q, safe="")

