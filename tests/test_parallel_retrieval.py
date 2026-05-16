"""Recuperación paralela (asyncio.gather) de fuentes y etapas."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from app.capabilities.evidence_rag.retrieval_parallel import (
    gather_sync_calls_blocking,
    parallel_retrieval_enabled,
    partial_call,
)
from app.capabilities.evidence_rag.multi_source_evidence_capability import (
    MultiSourceEvidenceCapability,
)
from app.schemas.copilot_state import ArticleSummary, EvidenceBundle


def test_parallel_retrieval_enabled_default(monkeypatch):
    monkeypatch.delenv("COPILOT_PARALLEL_RETRIEVAL", raising=False)
    assert parallel_retrieval_enabled() is True


def test_parallel_retrieval_disabled(monkeypatch):
    monkeypatch.setenv("COPILOT_PARALLEL_RETRIEVAL", "0")
    assert parallel_retrieval_enabled() is False


def test_gather_sync_calls_blocking_runs_in_parallel(monkeypatch):
    monkeypatch.setenv("COPILOT_RETRIEVAL_MAX_PARALLEL", "4")
    started: list[float] = []

    def slow(n: int) -> int:
        started.append(time.monotonic())
        time.sleep(0.08)
        return n

    t0 = time.monotonic()
    out = gather_sync_calls_blocking(
        [partial_call(slow, 1), partial_call(slow, 2), partial_call(slow, 3)],
        limit=3,
    )
    elapsed = time.monotonic() - t0
    assert out == [1, 2, 3]
    assert elapsed < 0.22


def test_multi_source_parallel_invokes_both_sources(monkeypatch):
    monkeypatch.setenv("COPILOT_PARALLEL_RETRIEVAL", "1")

    a = MagicMock()
    a.retrieve_evidence.return_value = EvidenceBundle(
        search_term="q",
        pmids=["1"],
        articles=[
            ArticleSummary(pmid="1", title="A", abstract_snippet="", year=2020),
        ],
    )
    b = MagicMock()
    b.retrieve_evidence.return_value = EvidenceBundle(
        search_term="q",
        pmids=["2"],
        articles=[
            ArticleSummary(pmid="2", title="B", abstract_snippet="", year=2021),
        ],
    )

    cap = MultiSourceEvidenceCapability(sources=[a, b])
    bundle = cap.retrieve_evidence("diabetes", retmax=6, years_back=5)
    assert set(bundle.pmids) == {"1", "2"}
    a.retrieve_evidence.assert_called_once()
    b.retrieve_evidence.assert_called_once()
