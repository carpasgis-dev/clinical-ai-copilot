"""
Implementación de ``EvidenceCapability`` sobre NCBI E-utilities (PRSN/pubmed).

La query la construye un ``EvidenceQueryPlanner`` (heurística, LLM o composite).
"""
from __future__ import annotations

from typing import Optional, Union

import httpx

from app.capabilities.evidence_rag.ncbi.eutils import esearch_pubmed, search_and_fetch_abstracts
from app.capabilities.evidence_rag.query_planning.protocol import EvidenceQueryPlanner
from app.capabilities.evidence_rag.query_planning.resolver import resolve_query_planner
from app.schemas.copilot_state import (
    ArticleSummary,
    ClinicalContext,
    EvidenceBundle,
    _ARTICLE_MAX_SNIPPET,
    _EVIDENCE_MAX_ART,
)


class NcbiEvidenceCapability:
    """
    Capability B con PubMed real (esearch + efetch), misma base que ``prsn30.pubmed.eutils``.

    Variables de entorno opcionales: ``NCBI_EMAIL``, ``NCBI_API_KEY`` (recomendado por NCBI).
    """

    def __init__(self, planner: EvidenceQueryPlanner | None = None) -> None:
        self._planner = planner if planner is not None else resolve_query_planner()

    def build_pubmed_query(
        self,
        free_text: str,
        clinical_context: Optional[Union[ClinicalContext, dict]] = None,
    ) -> str:
        return self._planner.build_query(free_text, clinical_context)

    def retrieve_evidence(
        self,
        pubmed_query: str,
        retmax: int = 6,
        years_back: int = 5,
    ) -> EvidenceBundle:
        term = (pubmed_query or "").strip()
        if not term:
            return EvidenceBundle(search_term="", pmids=[], articles=[])

        cap = min(max(retmax, 1), 10, _EVIDENCE_MAX_ART)
        yb: Optional[int] = years_back if years_back and years_back > 0 else None

        try:
            records = search_and_fetch_abstracts(
                term,
                retmax=cap,
                pubmed_years_back=yb,
            )
        except Exception:
            return EvidenceBundle(search_term=term, pmids=[], articles=[])

        articles: list[ArticleSummary] = []
        pmids: list[str] = []
        for rec in records[:_EVIDENCE_MAX_ART]:
            pmids.append(rec.pmid)
            y_int: Optional[int] = None
            if rec.year and str(rec.year).isdigit():
                y_int = int(str(rec.year)[:4])
            articles.append(
                ArticleSummary(
                    pmid=rec.pmid,
                    title=rec.title,
                    abstract_snippet=(rec.abstract or "")[:_ARTICLE_MAX_SNIPPET],
                    year=y_int,
                    doi=rec.doi,
                    open_access=None,
                )
            )

        return EvidenceBundle(
            search_term=term,
            pmids=pmids,
            articles=articles,
            chunks_used=0,
            oa_pdfs_retrieved=0,
        )

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=15.0) as client:
                esearch_pubmed("diabetes", retmax=1, client=client)
            return True
        except Exception:
            return False
