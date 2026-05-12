"""
Integración real con NCBI (opcional).

Ejecutar: ``set RUN_NCBI_INTEGRATION=1`` (Windows) o ``export RUN_NCBI_INTEGRATION=1`` (Unix)
y ``pytest tests/test_graph_pubmed_integration.py -v``.
"""
from __future__ import annotations

import os

import pytest

from app.orchestration.graph import build_copilot_graph
from app.schemas.copilot_state import Route


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_NCBI_INTEGRATION", "").strip() != "1",
    reason="Definir RUN_NCBI_INTEGRATION=1 para llamar a NCBI (red + NCBI_EMAIL recomendado).",
)
def test_hero_hybrid_invokes_real_pubmed() -> None:
    hero = (
        "Paciente con diabetes e hipertensión mayor de 65 años. "
        "¿Qué evidencia reciente existe sobre tratamientos que "
        "reduzcan riesgo cardiovascular?"
    )
    graph = build_copilot_graph()
    result = graph.invoke({"user_query": hero, "session_id": "e2e-ncbi-1"})

    assert result["route"] == Route.HYBRID
    assert result.get("pubmed_query")
    pmids = result.get("evidence_bundle", {}).get("pmids") or []
    assert pmids, "Se esperaban PMIDs reales desde PubMed"
    assert all(str(p).isdigit() for p in pmids)
