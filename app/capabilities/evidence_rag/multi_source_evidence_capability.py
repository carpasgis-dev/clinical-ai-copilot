"""
Capability de evidencia multi-fuente: NCBI + Europe PMC (u otras ``EvidenceCapability``).

Dedupe por PMID, fusiona metadatos, prioriza OA y ordena por año (rerank ligero).
El grafo y ``EvidenceBundle`` / ``ArticleSummary`` siguen siendo el contrato único.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from app.capabilities.contracts import EvidenceCapability
from app.capabilities.evidence_rag.europe_pmc import EuropePmcCapability
from app.capabilities.evidence_rag.ncbi_evidence_capability import NcbiEvidenceCapability
from app.capabilities.evidence_rag.query_planning.protocol import EvidenceQueryPlanner
from app.capabilities.evidence_rag.query_planning.resolver import resolve_query_planner
from app.schemas.copilot_state import (
    ArticleSummary,
    ClinicalContext,
    EvidenceBundle,
    _EVIDENCE_MAX_ART,
)


def _merge_open_access(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is True or b is True:
        return True
    if a is False or b is False:
        return False
    return None


def _merge_two_articles(x: ArticleSummary, y: ArticleSummary) -> ArticleSummary:
    """Fusiona dos resúmenes del mismo PMID (distintas fuentes)."""
    abstract = (
        x.abstract_snippet
        if len(x.abstract_snippet or "") >= len(y.abstract_snippet or "")
        else y.abstract_snippet
    )
    title = x.title if len(x.title or "") >= len(y.title or "") else y.title
    years = [v for v in (x.year, y.year) if v is not None]
    year = max(years) if years else None
    doi = x.doi or y.doi
    oa = _merge_open_access(x.open_access, y.open_access)
    return ArticleSummary(
        pmid=x.pmid,
        title=title,
        abstract_snippet=abstract,
        year=year,
        doi=doi,
        open_access=oa,
    )


def _oa_sort_rank(a: ArticleSummary) -> int:
    """Menor = más prioritario (OA explícito primero)."""
    if a.open_access is True:
        return 0
    if a.open_access is None:
        return 1
    return 2


def _dedupe_and_merge(articles: Sequence[ArticleSummary]) -> List[ArticleSummary]:
    by_pmid: Dict[str, ArticleSummary] = {}
    for art in articles:
        pmid = (art.pmid or "").strip()
        if not pmid:
            continue
        if pmid not in by_pmid:
            by_pmid[pmid] = art
        else:
            by_pmid[pmid] = _merge_two_articles(by_pmid[pmid], art)
    merged = list(by_pmid.values())
    merged.sort(key=lambda a: (_oa_sort_rank(a), -(a.year or 0), a.pmid))
    return merged


class MultiSourceEvidenceCapability:
    """
    Orquesta varias capabilities de evidencia y devuelve un único ``EvidenceBundle``.

    Por defecto: ``NcbiEvidenceCapability`` + ``EuropePmcCapability``.
    """

    def __init__(
        self,
        sources: Optional[Sequence[EvidenceCapability]] = None,
        planner: EvidenceQueryPlanner | None = None,
    ) -> None:
        self._planner = planner if planner is not None else resolve_query_planner()
        if sources is None:
            p = self._planner
            self._sources: List[EvidenceCapability] = [
                NcbiEvidenceCapability(planner=p),
                EuropePmcCapability(planner=p),
            ]
        else:
            self._sources = list(sources)

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
        combined: List[ArticleSummary] = []
        chunks = 0
        oa_count = 0

        for src in self._sources:
            try:
                b = src.retrieve_evidence(term, retmax=cap, years_back=years_back)
            except Exception:
                continue
            chunks += int(b.chunks_used or 0)
            oa_count += int(b.oa_pdfs_retrieved or 0)
            combined.extend(b.articles)

        merged = _dedupe_and_merge(combined)
        merged = merged[:_EVIDENCE_MAX_ART]
        oa_flags = sum(1 for a in merged if a.open_access is True)
        return EvidenceBundle(
            search_term=term,
            pmids=[a.pmid for a in merged],
            articles=merged,
            chunks_used=chunks,
            oa_pdfs_retrieved=max(oa_count, oa_flags),
        )

    def health_check(self) -> bool:
        if not self._sources:
            return False
        return any(s.health_check() for s in self._sources)
