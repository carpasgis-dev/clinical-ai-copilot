"""
Registro mínimo de artículo PubMed (fase abstract).

Origen: ``sina_mcp/prsn3.0/src/prsn30/models/pubmed_record.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PubMedArticleRecord:
    """Un artículo PubMed mínimo para RAG (fase 1: abstract)."""

    pmid: str
    title: str
    abstract: str
    year: Optional[str]
    doi: Optional[str]
