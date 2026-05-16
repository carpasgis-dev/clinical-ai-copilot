"""Síntesis narrativa opcional (``COPILOT_SYNTHESIS=llm``)."""
from __future__ import annotations

import pytest


def test_synthesis_uses_llm_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COPILOT_SYNTHESIS", raising=False)
    from app.orchestration.llm_synthesis import synthesis_uses_llm

    assert synthesis_uses_llm() is False


def test_synthesis_uses_llm_requires_mode_and_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_SYNTHESIS", "llm")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "local")
    from app.orchestration.llm_synthesis import synthesis_uses_llm

    assert synthesis_uses_llm() is True

    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert synthesis_uses_llm() is False


def test_try_llm_synthesis_fallback_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_SYNTHESIS", "llm")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    from app.orchestration import llm_synthesis as ls

    def _boom(**_kwargs: object) -> str:
        raise RuntimeError("no server")

    monkeypatch.setattr(ls, "_openai_chat_completion_text", _boom)
    text, warns = ls.try_llm_synthesis_narrative(
        {"user_query": "test", "route": "sql"},
        {"summary": "s", "citations": []},
    )
    assert text is None
    assert warns and "synthesis_llm" in warns[0]


def test_try_llm_synthesis_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_SYNTHESIS", "llm")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "m")

    from app.orchestration import llm_synthesis as ls

    captured: dict[str, str] = {}

    def _fake_chat(**kw: object) -> str:
        captured["user"] = str(kw.get("user") or "")
        return "  Respuesta redactada por el modelo.  "

    monkeypatch.setattr(ls, "_openai_chat_completion_text", _fake_chat)
    text, warns = ls.try_llm_synthesis_narrative(
        {"user_query": "tratamientos que reduzcan riesgo cardiovascular"},
        {"summary": "x", "citations": []},
    )
    assert text == "Respuesta redactada por el modelo."
    assert not warns
    assert "HECHOS_JSON" in captured["user"]
    assert "Borrador" not in captured["user"]
    assert "Hallazgos clave" not in captured["user"]
    assert "Síntesis sobre la pregunta clínica" in captured["user"]
    assert "comparación solicitada" not in captured["user"].lower()


def test_question_requests_direct_therapeutic_comparison() -> None:
    from app.orchestration.llm_synthesis import question_requests_direct_therapeutic_comparison

    q = (
        "¿Qué dice la evidencia sobre anticoagulantes orales directos frente a warfarina "
        "para prevención de ictus?"
    )
    assert question_requests_direct_therapeutic_comparison(q) is True
    assert question_requests_direct_therapeutic_comparison("¿Cuántos pacientes con diabetes?") is False


def test_compact_facts_dedupes_pmids() -> None:
    from app.orchestration.llm_synthesis import _compact_facts_json

    raw = _compact_facts_json(
        {"user_query": "q", "route": "hybrid"},
        {
            "citations": [
                {"pmid": "99", "title": "First"},
                {"pmid": "99", "title": "Dup"},
                {"pmid": "100", "title": "Second"},
            ],
        },
    )
    data = __import__("json").loads(raw)
    assert data["num_pmids_unicos"] == 2
    assert len(data["citas_pubmed"]) == 2


def test_dedupe_pmid_sections_keeps_first_block() -> None:
    from app.orchestration.llm_synthesis import dedupe_pmid_sections

    text = (
        "## Evidencia\n\n"
        "### PMID 42050587 — Trial A\nPrimera versión.\n\n"
        "### PMID 42057665 — Trial B\nOk.\n\n"
        "### PMID 42050587 — Trial A again\nDuplicado.\n"
    )
    out = dedupe_pmid_sections(text)
    assert out.count("### PMID 42050587") == 1
    assert "Primera versión" in out
    assert "Duplicado" not in out


def test_dedupe_pmid_sections_malformed_repetido_heading() -> None:
    from app.orchestration.llm_synthesis import dedupe_pmid_sections

    text = (
        "### PMID 41363914 — PCOS study\nPrimera.\n\n"
        "### PMID 41363914 (repetido, pero con un extracto diferente)\n"
        + "Bucle.\n" * 5
        + "Síntesis sobre la pregunta clínica\nCierre ok.\n"
    )
    out = dedupe_pmid_sections(text)
    assert out.count("### PMID 41363914") == 1
    assert "Bucle" not in out
    assert "Síntesis sobre la pregunta clínica" in out


def test_cv_risk_question_not_flagged_as_pairwise_comparison() -> None:
    from app.orchestration.llm_synthesis import question_requests_direct_therapeutic_comparison

    q = (
        "Paciente con diabetes e hipertensión mayor de 65 años. "
        "¿Qué evidencia reciente existe sobre tratamientos que reduzcan riesgo cardiovascular?"
    )
    assert question_requests_direct_therapeutic_comparison(q) is False


def test_compact_facts_flags_comparison_for_hybrid_question() -> None:
    from app.orchestration.llm_synthesis import _compact_facts_json

    uq = "DOAC vs warfarina en fibrilación auricular"
    raw = _compact_facts_json(
        {"user_query": uq, "route": "hybrid"},
        {
            "summary": "n=0",
            "cohort_size": 0,
            "citations": [{"pmid": "1", "title": "Trial A vs B"}],
        },
    )
    data = __import__("json").loads(raw)
    assert data["pregunta_pide_comparacion_terapeutica_directa"] is True
    assert data["citas_pubmed"][0]["pmid"] == "1"


def test_medical_answer_after_llm_updates_limitations() -> None:
    from app.orchestration.llm_synthesis import medical_answer_after_llm_synthesis

    ma = {
        "limitations": [
            "Síntesis generada con reglas deterministas (no LLM clínico); no sustituye juicio clínico.",
            "Verificar siempre frente a fuentes primarias y protocolos del centro.",
        ]
    }
    out = medical_answer_after_llm_synthesis(ma)
    lims = out["limitations"]
    assert not any("reglas deterministas (no LLM" in x for x in lims)
    assert any("LLM" in x and "OpenAI" in x for x in lims)
