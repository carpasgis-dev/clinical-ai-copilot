"""Planificador de query vía LLM (OpenAI-compatible) + post-proceso PRSN-style."""
from __future__ import annotations

import json
import os
from typing import Any, Optional, Union
from urllib.parse import urlparse

import httpx

from app.capabilities.evidence_rag.copilot_errors import CopilotError
from app.capabilities.evidence_rag.heuristic_evidence_query import (
    as_clinical_context,
    build_evidence_search_query,
)
from app.capabilities.evidence_rag.query_planning.llm_postprocess import (
    _SYSTEM,
    clip_text,
    env_int,
    finalize_llm_pubmed_line,
    is_spanish_pubmed_line,
    pubmed_line_from_llm_text,
)
from app.schemas.copilot_state import ClinicalContext


def normalize_openai_compatible_base_url(raw: str) -> str:
    """
    Normaliza ``LLM_BASE_URL`` para ``POST {base}/chat/completions``.

    Si la URL no tiene ruta (solo host:puerto), añade ``/v1`` como hacen Ollama,
    llama-server y LM Studio. No modifica rutas ya explícitas (p. ej. ``/proxy/v1``).
    """
    b = (raw or "").strip().rstrip("/")
    if not b:
        return b
    if b.lower().endswith("/v1"):
        return b
    parsed = urlparse(b)
    path = (parsed.path or "").rstrip("/")
    if path in ("", "/"):
        return f"{b}/v1"
    return b


def _compact_demographics_block(
    clinical_context: Optional[Union[ClinicalContext, dict]],
) -> str:
    """Solo señales mínimas de edad/sexo para el prompt (evita copiar listas largas de diagnósticos)."""
    ctx = as_clinical_context(clinical_context)
    if ctx is None:
        return ""
    max_prof = env_int("COPILOT_PUBMED_PLANNER_MAX_PROFILE", 600)
    lines: list[str] = []
    if ctx.population_age_min is not None and ctx.population_age_min >= 65:
        lines.append(f"Population: age >= {ctx.population_age_min} (older adults)")
    elif ctx.population_age_max is not None and ctx.population_age_max <= 18:
        lines.append("Population: pediatric age filter")
    else:
        ar = (ctx.age_range or "").strip().lower()
        if ar:
            if "65" in ar or "ancian" in ar or "anciano" in ar:
                lines.append("Patient: elderly")
            elif "<18" in ar or "pediatric" in ar or "niño" in ar or "nino" in ar:
                lines.append("Patient: pediatric")
            else:
                lines.append(f"Patient age context: {ctx.age_range}")
    if (ctx.population_sex or "").strip().upper() == "F":
        lines.append("Population: female")
    elif (ctx.population_sex or "").strip().upper() == "M":
        lines.append("Population: male")
    if ctx.population_conditions:
        lines.append("Population conditions (cohort): " + ", ".join(ctx.population_conditions[:8]))
    if ctx.population_medications:
        lines.append("Population medications (cohort): " + ", ".join(ctx.population_medications[:8]))
    if ctx.population_size is not None:
        lines.append(f"Local cohort size: {ctx.population_size}")
    return clip_text("\n".join(lines), max_prof)


def _openai_chat_completion_text(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> str:
    base = normalize_openai_compatible_base_url(base_url or "")
    if not base:
        raise RuntimeError("LLM_BASE_URL vacío")
    url = f"{base}/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t = max(5.0, float(timeout_s))
    connect_s = min(45.0, max(5.0, t * 0.2))
    timeout_cfg = httpx.Timeout(connect=connect_s, read=t, write=min(t, 180.0), pool=10.0)
    with httpx.Client(timeout=timeout_cfg) as client:
        r = client.post(url, headers=headers, content=json.dumps(payload))
        r.raise_for_status()
        data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    return (content or "").strip() if isinstance(content, str) else ""


class LlmQueryPlanner:
    """
    Una llamada chat al endpoint OpenAI-compatible (``LLM_BASE_URL`` + ``LLM_MODEL``).

    Fallos (red, vacío, salida en español, post-proceso): ``CopilotError`` con ``code``.
    """

    def __init__(
        self,
        *,
        max_out_tokens: int | None = None,
        temperature_cap: float = 0.15,
        timeout_s: float | None = None,
    ) -> None:
        self._max_out = max_out_tokens or env_int("COPILOT_PUBMED_PLANNER_MAX_TOKENS", 384)
        self._temp_cap = temperature_cap
        self._timeout = timeout_s or float(os.getenv("COPILOT_PUBMED_PLANNER_TIMEOUT", "60"))

    def build_query(
        self,
        free_text: str,
        clinical_context: Optional[Union[ClinicalContext, dict]] = None,
    ) -> str:
        q_es = " ".join((free_text or "").strip().split())
        if not q_es:
            return ""

        max_q = env_int("COPILOT_PUBMED_PLANNER_MAX_QUESTION", 550)
        max_hint = env_int("COPILOT_PUBMED_PLANNER_MAX_HINT", 750)
        q_clip = clip_text(q_es, max_q)
        prof = _compact_demographics_block(clinical_context)
        hint = clip_text(build_evidence_search_query(q_es, clinical_context), max_hint)

        user_body = (
            "Translate the clinical question into ONE English PubMed search line (AND/OR, quoted phrases).\n\n"
            f"SPANISH CLINICAL QUESTION:\n{q_clip or '(none)'}\n\n"
            f"Patient demographics (age/sex filters only if relevant):\n{prof or '(none)'}\n\n"
            f"English vocabulary hint (adapt/simplify this; keep conditions, intervention, outcome):\n"
            f"{hint or 'clinical evidence'}\n\n"
            "RULES: English only. Max 3 AND groups. Do not copy Spanish words. "
            "If the hint lists drug classes (SGLT2, GLP-1), join them with OR in one group; "
            "do NOT require outcome terms (MACE, mortality) in the same query unless the hint "
            "is outcome-only with no drug class. Comparator drugs (e.g. metformin) are optional, not mandatory AND. "
            "Prefer MeSH Terms OR tiab for diseases and drug classes. "
            "Example:\n"
            '("atrial fibrillation") AND ("direct oral anticoagulant" OR warfarin) AND ("stroke prevention")'
        )

        base = os.getenv("LLM_BASE_URL", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        if not base or not model:
            raise CopilotError(
                "PUBMED_PLANNER_MISSING_LLM_CONFIG",
                "Falta LLM_BASE_URL o LLM_MODEL para LlmQueryPlanner",
            )

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
        temp = min(float(os.getenv("LLM_TEMPERATURE", "0.1")), self._temp_cap)

        try:
            raw = _openai_chat_completion_text(
                base_url=base,
                api_key=api_key,
                model=model,
                system=_SYSTEM,
                user=user_body,
                max_tokens=self._max_out,
                temperature=temp,
                timeout_s=self._timeout,
            )
        except Exception as exc:
            raise CopilotError(
                "PUBMED_PLANNER_LLM_REQUEST_FAILED",
                f"Llamada al LLM falló: {exc}",
                cause=exc,
            ) from exc
        if not raw.strip():
            raise CopilotError("PUBMED_PLANNER_LLM_EMPTY_RESPONSE", "LLM devolvió respuesta vacía")

        line0 = pubmed_line_from_llm_text(raw)
        if is_spanish_pubmed_line(line0):
            raise CopilotError(
                "PUBMED_PLANNER_SPANISH_OUTPUT",
                "El LLM devolvió una línea PubMed en español (no se reintenta).",
            )

        out = finalize_llm_pubmed_line(line0)
        if not out.strip():
            raise CopilotError(
                "PUBMED_PLANNER_POSTPROCESS_EMPTY",
                "Query PubMed vacía tras post-proceso",
            )
        return out
