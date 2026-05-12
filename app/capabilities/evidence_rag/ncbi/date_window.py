"""Ventana de fechas de publicación (PDAT) para NCBI esearch.

Origen: ``sina_mcp/prsn3.0/src/prsn30/pubmed/date_window.py``.
"""
from __future__ import annotations

from datetime import date
from typing import Optional, Tuple


def pdat_range_for_years_back(
    years_back: Optional[int],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Devuelve (mindate, maxdate) en formato YYYY/MM/DD para ``datetype=pdat``.

    Si ``years_back`` es None o <= 0, no se aplica filtro (None, None).
    """
    if years_back is None or years_back <= 0:
        return None, None
    today = date.today()
    start_year = today.year - int(years_back)
    mindate = date(start_year, 1, 1).strftime("%Y/%m/%d")
    maxdate = today.strftime("%Y/%m/%d")
    return mindate, maxdate


def describe_pdat_filter(years_back: Optional[int]) -> Optional[str]:
    """Texto corto para logs / UI; None si no hay filtro."""
    if years_back is None or years_back <= 0:
        return None
    mindate, maxdate = pdat_range_for_years_back(years_back)
    if not mindate or not maxdate:
        return None
    return (
        f"Publicación PubMed (PDAT): {mindate} — {maxdate} (~últimos {years_back} años)"
    )
