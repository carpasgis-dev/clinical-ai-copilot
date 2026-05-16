
"""
Cliente Europe PMC REST + ``EvidenceCapability``.

API: https://www.ebi.ac.uk/europepmc/webservices/rest/search

El grafo y los DTOs no cambian: se devuelve ``EvidenceBundle`` / ``ArticleSummary``.
"""
from __future__ import annotations

from app.config.settings import settings

from datetime import date
from typing import Any, Dict, List, Optional, Union

import httpx

from app.capabilities.evidence_rag.query_planning.protocol import EvidenceQueryPlanner
from app.capabilities.evidence_rag.query_planning.resolver import resolve_query_planner
from app.schemas.copilot_state import (
    ArticleSummary,
    ClinicalContext,
    EvidenceBundle,
)

EUROPE_PMC_SEARCH = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
)


def _epmc_retrieval_debug(
    *,
    outcome: str,
    term: str = "",
    errors: list[str] | None = None,
    articles_parsed: int = 0,
) -> dict[str, Any]:
    from app.capabilities.evidence_rag.ncbi.pubmed_query_normalizer import (
        retrieval_metrics_for_query,
    )

    return {
        "outcome": outcome,
        "backend": "europe_pmc",
        "attempts": [],
        "errors": list(errors or []),
        "final_idlist_length": articles_parsed,
        "articles_parsed": articles_parsed,
        "pubmed_query_planned": term,
        "normalized_query": term,
        "final_query_sent": term,
        "retrieval_metrics": retrieval_metrics_for_query(term),
    }


def _pdat_filter_clause(years_back: int) -> str:
    """Fragmento de query Europe PMC para ventana de publicación (FIRST_PDATE)."""
    if years_back <= 0:
        return ""
    today = date.today()
    start = date(today.year - int(years_back), 1, 1)
    # Sintaxis rango: https://europepmc.org/Help
    return (
        f' FIRST_PDATE:[{start.strftime("%Y-%m-%d")} TO '
        f'{today.strftime("%Y-%m-%d")}]'
    )


def _normalize_pmid(hit: Dict[str, Any]) -> str:
    raw = hit.get("pmid")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    rid = str(hit.get("id") or "").strip()
    if rid.upper().startswith("MED"):
        # ej. MED:12345678 o 12345678
        parts = rid.replace("MED:", "").replace("med:", "").split("/")
        return parts[-1].strip()
    if rid.isdigit():
        return rid
    return ""


def _parse_year(hit: Dict[str, Any]) -> Optional[int]:
    y = hit.get("pubYear") or hit.get("year")
    if y is None:
        return None
    s = str(y).strip()[:4]
    return int(s) if s.isdigit() else None


def _hit_open_access(hit: Dict[str, Any]) -> Optional[bool]:
    v = hit.get("isOpenAccess")
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().upper()
    if s in ("Y", "YES", "TRUE", "1"):
        return True
    if s in ("N", "NO", "FALSE", "0"):
        return False
    return None


def search_europe_pmc(
    query: str,
    *,
    page_size: int = 6,
    years_back: int = 0,
    client: Optional[httpx.Client] = None,
) -> List[Dict[str, Any]]:
    """
    Llama a ``/search`` y devuelve la lista ``result`` (dicts crudos).

    ``query`` ya es el término completo (puede incluir FIRST_PDATE si aplica).
    """
    q = (query or "").strip()
    if not q:
        return []

    params: Dict[str, Any] = {
        "query": q + _pdat_filter_clause(years_back),
        "format": "json",
        "pageSize": min(max(page_size, 1), 25),
        "resultType": "core",
        "src": "MED",
    }
    own = client is None
    if own:
        client = httpx.Client(timeout=60.0)
    try:
        assert client is not None
        r = client.get(EUROPE_PMC_SEARCH, params=params)
        r.raise_for_status()
        data = r.json()
        rlist = data.get("resultList") or {}
        results = rlist.get("result")
        if not results:
            return []
        if isinstance(results, dict):
            return [results]
        return list(results)
    finally:
        if own and client is not None:
            client.close()


async def search_europe_pmc_async(
    query: str,
    *,
    page_size: int = 6,
    years_back: int = 0,
    client: Optional[httpx.AsyncClient] = None,
) -> List[Dict[str, Any]]:
    """Versión async de ``search_europe_pmc`` para ``asyncio.gather``."""
    q = (query or "").strip()
    if not q:
        return []

    params: Dict[str, Any] = {
        "query": q + _pdat_filter_clause(years_back),
        "format": "json",
        "pageSize": min(max(page_size, 1), 25),
        "resultType": "core",
        "src": "MED",
    }
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=60.0)
    try:
        assert client is not None
        r = await client.get(EUROPE_PMC_SEARCH, params=params)
        r.raise_for_status()
        data = r.json()
        rlist = data.get("resultList") or {}
        results = rlist.get("result")
        if not results:
            return []
        if isinstance(results, dict):
            return [results]
        return list(results)
    finally:
        if own and client is not None:
            await client.aclose()


class EuropePmcCapability:
    """
    Misma superficie que ``NcbiEvidenceCapability``: solo cambia la capa HTTP/API.

    Filtro ``src=MED`` para alinear con literatura tipo PubMed; se puede ampliar
    (PMC, PPR, etc.) sin tocar el grafo.
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
        *,
        synthesis_pubtype_refine: bool = True,
    ) -> EvidenceBundle:
        term = (pubmed_query or "").strip()
        if not term:
            return EvidenceBundle(
                search_term="",
                pmids=[],
                articles=[],
                retrieval_debug=_epmc_retrieval_debug(
                    outcome="no_query",
                    errors=["término de búsqueda vacío"],
                ),
            )

        cap = min(max(retmax, 1), 10, settings.evidence_max_art)
        yb = years_back if years_back and years_back > 0 else 0

        try:
            from app.capabilities.evidence_rag.retrieval_parallel import (
                parallel_retrieval_enabled,
                run_coroutine_sync,
            )

            if parallel_retrieval_enabled():
                hits = run_coroutine_sync(
                    search_europe_pmc_async(term, page_size=cap, years_back=yb)
                )
            else:
                hits = search_europe_pmc(
                    term,
                    page_size=cap,
                    years_back=yb,
                )
        except Exception as exc:
            return EvidenceBundle(
                search_term=term,
                pmids=[],
                articles=[],
                retrieval_debug=_epmc_retrieval_debug(
                    outcome="error",
                    term=term,
                    errors=[str(exc)],
                ),
            )

        articles: list[ArticleSummary] = []
        pmids: list[str] = []
        for hit in hits[:settings.evidence_max_art]:
            pmid = _normalize_pmid(hit)
            if not pmid or not pmid.isdigit():
                continue
            title = str(hit.get("title") or "(sin título)")[:2000]
            abstract = str(hit.get("abstractText") or hit.get("abstract") or "")
            doi = hit.get("doi")
            doi_s = str(doi).strip() if doi else None
            articles.append(
                ArticleSummary(
                    pmid=pmid,
                    title=title,
                    abstract_snippet=abstract[:settings.article_max_snippet],
                    year=_parse_year(hit),
                    doi=doi_s,
                    open_access=_hit_open_access(hit),
                )
            )
            pmids.append(pmid)

        return EvidenceBundle(
            search_term=term,
            pmids=pmids,
            articles=articles,
            chunks_used=0,
            oa_pdfs_retrieved=0,
            retrieval_debug=_epmc_retrieval_debug(
                outcome="ok",
                term=term,
                articles_parsed=len(articles),
            ),
        )

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=15.0) as client:
                search_europe_pmc("diabetes", page_size=1, years_back=0, client=client)
            return True
        except Exception:
            return False
