"""
Capability de evidencia determinista para tests y CI (sin red).

Replica el comportamiento de los stubs previos del grafo para que
``pytest`` no dependa de NCBI.
"""
from __future__ import annotations

from typing import Optional, Union

from app.capabilities.evidence_rag.heuristic_evidence_query import build_evidence_search_query
from app.capabilities.evidence_rag.ncbi.pubmed_query_normalizer import (
    normalize_pubmed_query,
    retrieval_metrics_for_query,
)
from app.schemas.copilot_state import (
    ArticleSummary,
    ClinicalContext,
    EvidenceBundle,
)


class StubEvidenceCapability:
    """Misma API que ``NcbiEvidenceCapability``; datos ficticios acotados."""

    def build_pubmed_query(
        self,
        free_text: str,
        clinical_context: Optional[Union[ClinicalContext, dict]] = None,
    ) -> str:
        """Misma heurística que el planner compartido (cohorte + texto)."""
        return build_evidence_search_query(free_text, clinical_context)

    def retrieve_evidence(
        self,
        pubmed_query: str,
        retmax: int = 6,
        years_back: int = 5,
        *,
        synthesis_pubtype_refine: bool = True,
    ) -> EvidenceBundle:
        term = (pubmed_query or "").strip()[:200] or "(empty)"
        planned = (pubmed_query or "").strip()
        nq, nm = normalize_pubmed_query(planned)
        metrics = retrieval_metrics_for_query(nq)
        dbg_tail = {
            "pubmed_query_planned": planned,
            "normalized_query": nq or planned,
            "final_query_sent": nq or planned,
            "normalization": nm,
            "retrieval_metrics": metrics,
        }
        # Ruta híbrida: la query incluye términos MeSH típicos del builder.
        if "[MeSH Terms]" in pubmed_query or "[tiab]" in pubmed_query.lower():
            return EvidenceBundle(
                search_term=term,
                pmids=["00000002", "00000003"],
                articles=[
                    ArticleSummary(
                        pmid="00000002",
                        title="Stub article A (hybrid)",
                        abstract_snippet="Snippet A.",
                        year=2023,
                    ),
                    ArticleSummary(
                        pmid="00000003",
                        title="Stub article B (hybrid)",
                        abstract_snippet="Snippet B.",
                        year=2022,
                    ),
                ],
                retrieval_debug={
                    "outcome": "stub",
                    "backend": "stub_evidence",
                    "attempts": [],
                    "errors": [],
                    "final_idlist_length": 2,
                    "articles_parsed": 2,
                    **dbg_tail,
                },
            )
        return EvidenceBundle(
            search_term=term,
            pmids=["00000001"],
            articles=[
                ArticleSummary(
                    pmid="00000001",
                    title="Stub article (evidence route)",
                    abstract_snippet="Abstract stub.",
                    year=2024,
                )
            ],
            retrieval_debug={
                "outcome": "stub",
                "backend": "stub_evidence",
                "attempts": [],
                "errors": [],
                "final_idlist_length": 1,
                "articles_parsed": 1,
                **dbg_tail,
            },
        )

    def health_check(self) -> bool:
        return True
