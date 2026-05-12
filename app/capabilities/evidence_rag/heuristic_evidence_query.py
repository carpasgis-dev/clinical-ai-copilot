"""
Query de búsqueda de evidencia en texto libre (heurística compartida).

Misma cadena sirve como término para NCBI E-utilities y como ``query`` para
Europe PMC REST (sintaxis cercana; afinar por backend si hace falta).
"""
from __future__ import annotations

from typing import Optional, Union

from app.schemas.copilot_state import ClinicalContext


def as_clinical_context(
    clinical_context: Optional[Union[ClinicalContext, dict]],
) -> Optional[ClinicalContext]:
    if clinical_context is None:
        return None
    if isinstance(clinical_context, ClinicalContext):
        return clinical_context
    return ClinicalContext.model_validate(clinical_context)


def build_evidence_search_query(
    free_text: str,
    clinical_context: Optional[Union[ClinicalContext, dict]] = None,
) -> str:
    """Combina texto del usuario con señales del contexto clínico (AND/OR simple)."""
    core = " ".join((free_text or "").strip().split())
    if not core:
        return ""

    ctx = as_clinical_context(clinical_context)
    if ctx is None:
        return core

    groups: list[str] = []
    if ctx.conditions:
        ors = " OR ".join(c.strip() for c in ctx.conditions if c.strip())
        if ors:
            groups.append(f"({ors})")
    if ctx.medications:
        meds = " OR ".join(m.strip() for m in ctx.medications if m.strip())
        if meds:
            groups.append(f"({meds})")
    if ctx.age_range and (
        "65" in ctx.age_range
        or ">" in ctx.age_range
        or "ancian" in ctx.age_range.lower()
    ):
        groups.append("(aged[MeSH Terms] OR elderly[MeSH Terms])")

    if not groups:
        return core
    joined = " AND ".join(groups)
    return f"({joined}) AND ({core})"
