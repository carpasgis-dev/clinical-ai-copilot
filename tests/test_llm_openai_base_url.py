"""Normalización de ``LLM_BASE_URL`` para clientes OpenAI-compatibles."""
from __future__ import annotations

from app.capabilities.evidence_rag.query_planning.llm_planner import normalize_openai_compatible_base_url


def test_normalize_appends_v1_for_host_only() -> None:
    assert normalize_openai_compatible_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434/v1"
    assert normalize_openai_compatible_base_url("http://localhost:8080") == "http://localhost:8080/v1"
    assert normalize_openai_compatible_base_url("https://api.openai.com") == "https://api.openai.com/v1"


def test_normalize_preserves_explicit_v1_or_custom_path() -> None:
    assert normalize_openai_compatible_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080/v1"
    assert normalize_openai_compatible_base_url("http://proxy.internal/openai/v1") == "http://proxy.internal/openai/v1"


def test_normalize_strips_and_handles_empty() -> None:
    assert normalize_openai_compatible_base_url("  http://x:1/v1/  ") == "http://x:1/v1"
    assert normalize_openai_compatible_base_url("") == ""
    assert normalize_openai_compatible_base_url("   ") == ""
