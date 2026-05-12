"""
Capability de evidencia determinista para tests y CI (sin red).

Replica el comportamiento de los stubs previos del grafo para que
``pytest`` no dependa de NCBI.
"""
from __future__ import annotations

from typing import Optional, Union

from app.schemas.copilot_state import (
    ArticleSummary,
    ClinicalContext,
    EvidenceBundle,
)


def _conditions_from_context(
    clinical_context: Optional[Union[ClinicalContext, dict]],
) -> list[str]:
    if clinical_context is None:
        return []
    if isinstance(clinical_context, ClinicalContext):
        return list(clinical_context.conditions or [])
    return list((clinical_context or {}).get("conditions") or [])


class StubEvidenceCapability:
    """Misma API que ``NcbiEvidenceCapability``; datos ficticios acotados."""

    def build_pubmed_query(
        self,
        free_text: str,
        clinical_context: Optional[Union[ClinicalContext, dict]] = None,
    ) -> str:
        conditions = _conditions_from_context(clinical_context)
        core = " ".join((free_text or "").strip().split())
        if not conditions:
            return core
        mesh = " AND ".join(f'("{c}"[MeSH Terms])' for c in conditions)
        return f"({mesh}) AND (therapy[tiab] OR treatment[tiab])"

    def retrieve_evidence(
        self,
        pubmed_query: str,
        retmax: int = 6,
        years_back: int = 5,
    ) -> EvidenceBundle:
        term = (pubmed_query or "").strip()[:200] or "(empty)"
        # Ruta híbrida: la query stub incluye [MeSH Terms].
        if "[MeSH Terms]" in pubmed_query:
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
        )

    def health_check(self) -> bool:
        return True
