"""Tests del normalizador PubMed-safe (sin red)."""
from __future__ import annotations

from app.capabilities.evidence_rag.ncbi.pubmed_query_normalizer import (
    normalize_pubmed_query,
    retrieval_metrics_for_query,
)


def test_or_mix_fielded_unfielded_gets_tiab_on_bare_quote() -> None:
    raw = '("diabetes mellitus"[tiab] OR "diabetic")'
    norm, meta = normalize_pubmed_query(raw)
    assert '"diabetic"[tiab]' in norm
    assert "tag_unfielded_quoted_phrases" in meta["steps_applied"]


def test_smart_quotes_to_ascii() -> None:
    raw = "\u201ccancer screening\u201d"
    norm, meta = normalize_pubmed_query(raw)
    assert norm.startswith('"cancer screening"')
    assert "sanitize_unicode" in meta["steps_applied"]


def test_collapse_duplicate_and() -> None:
    norm, _ = normalize_pubmed_query("metformin AND AND diabetes")
    assert "AND AND" not in norm
    assert "metformin AND diabetes" in norm


def test_balance_parentheses_adds_closing() -> None:
    norm, meta = normalize_pubmed_query("(a AND b")
    assert norm.endswith(")")
    assert any("equilibrio" in w for w in meta["warnings"])


def test_year_four_digits_not_tagged() -> None:
    norm, _ = normalize_pubmed_query('"2020"')
    assert norm == '"2020"'
    assert "[tiab]" not in norm


def test_retrieval_metrics_shape() -> None:
    q = '("a"[tiab]) AND ("b"[tiab]) AND ("c"[tiab]) AND ("d"[tiab])'
    norm, _ = normalize_pubmed_query(q)
    m = retrieval_metrics_for_query(norm)
    assert 0.0 <= m["query_complexity"] <= 1.0
    assert m["boolean_depth"] >= 1
    assert m["estimated_specificity"] in ("low", "medium", "high", "unknown")
    assert "operator_counts" in m


def test_skips_tiab_when_quoted_inner_already_contains_bracket_field() -> None:
    """No re-etiquetar frases que ya llevan ``[...]`` dentro de las comillas (evita corrupción)."""
    raw = '"[tiab]hypertension"'
    norm, _ = normalize_pubmed_query(raw)
    assert norm == raw
    assert '"[tiab]hypertension"[tiab]' not in norm
