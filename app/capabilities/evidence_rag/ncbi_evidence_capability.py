
"""
Implementación de ``EvidenceCapability`` sobre NCBI E-utilities (PRSN/pubmed).

La query la construye un ``EvidenceQueryPlanner`` (heurística, LLM o composite).
"""
from __future__ import annotations

from app.config.settings import settings

from typing import Optional, Union

import httpx

from app.capabilities.evidence_rag.ncbi.eutils import esearch_pubmed
from app.capabilities.evidence_rag.ncbi.eutils_async import search_and_fetch_parallel_aware
from app.capabilities.evidence_rag.ncbi.pubmed_query_normalizer import retrieval_metrics_for_query
from app.capabilities.evidence_rag.query_planning.protocol import EvidenceQueryPlanner
from app.capabilities.evidence_rag.query_planning.resolver import resolve_query_planner
from app.schemas.copilot_state import (
    ArticleSummary,
    ClinicalContext,
    EvidenceBundle,
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
            return EvidenceBundle(
                search_term="",
                pmids=[],
                articles=[],
                retrieval_debug={
                    "outcome": "no_query",
                    "attempts": [],
                    "errors": ["término PubMed vacío"],
                    "final_idlist_length": 0,
                    "articles_parsed": 0,
                    "pubmed_query_planned": "",
                    "normalized_query": "",
                    "final_query_sent": "",
                    "normalization": {"warnings": [], "steps_applied": []},
                    "retrieval_metrics": retrieval_metrics_for_query(""),
                },
            )

        fetch_cap = max(
            1,
            min(max(int(retmax), 1), int(settings.evidence_retrieval_pool_max)),
        )
        yb: Optional[int] = years_back if years_back and years_back > 0 else None

        records, dbg = search_and_fetch_parallel_aware(
            term,
            retmax=fetch_cap,
            pubmed_years_back=yb,
        )

        articles: list[ArticleSummary] = []
        pmids: list[str] = []
        for rec in records[:fetch_cap]:
            pmids.append(rec.pmid)
            y_int: Optional[int] = None
            if rec.year and str(rec.year).isdigit():
                y_int = int(str(rec.year)[:4])
            articles.append(
                ArticleSummary(
                    pmid=rec.pmid,
                    title=rec.title,
                    abstract_snippet=(rec.abstract or "")[:settings.article_max_snippet],
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
            retrieval_debug=dbg,
        )

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=15.0) as client:
                esearch_pubmed("diabetes", retmax=1, client=client)
            return True
        except Exception:
            return False
