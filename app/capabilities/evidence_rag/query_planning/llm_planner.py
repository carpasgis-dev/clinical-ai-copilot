"""Planificador de query vía LLM (OpenAI-compatible) + post-proceso PRSN-style."""
from __future__ import annotations

import json
import os
from typing import Any, Optional, Union

import httpx

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


def _compact_demographics_block(
    clinical_context: Optional[Union[ClinicalContext, dict]],
) -> str:
    """Solo señales mínimas de edad/sexo para el prompt (evita copiar listas largas de diagnósticos)."""
    ctx = as_clinical_context(clinical_context)
    if ctx is None:
        return ""
    max_prof = env_int("COPILOT_PUBMED_PLANNER_MAX_PROFILE", 600)
    lines: list[str] = []
    ar = (ctx.age_range or "").strip().lower()
    if ar:
        if "65" in ar or "ancian" in ar or "anciano" in ar:
            lines.append("Patient: elderly")
        elif "<18" in ar or "pediatric" in ar or "niño" in ar or "nino" in ar:
            lines.append("Patient: pediatric")
        else:
            lines.append(f"Patient age context: {ctx.age_range}")
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
    base = (base_url or "").rstrip("/")
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
    with httpx.Client(timeout=timeout_s) as client:
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

    Errores de red o respuesta vacía: ``build_query`` lanza ``RuntimeError`` para que
    ``CompositeQueryPlanner`` pueda hacer fallback heurístico.
    """

    def __init__(
        self,
        *,
        max_out_tokens: int | None = None,
        temperature_cap: float = 0.15,
        timeout_s: float | None = None,
    ) -> None:
        self._max_out = max_out_tokens or env_int("COPILOT_PUBMED_PLANNER_MAX_TOKENS", 128)
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
        max_hint = env_int("COPILOT_PUBMED_PLANNER_MAX_HINT", 300)
        q_clip = clip_text(q_es, max_q)
        prof = _compact_demographics_block(clinical_context)
        hint = clip_text(build_evidence_search_query(q_es, clinical_context), max_hint)

        user_body = (
            "Translate the clinical concepts from the question below into an English PubMed search string.\n\n"
            f"SPANISH CLINICAL QUESTION:\n{q_clip or '(none)'}\n\n"
            f"Patient demographics (use ONLY for age/sex filters if relevant):\n{prof or '(none)'}\n\n"
            f"English vocabulary hint (reference only):\n{hint or 'clinical evidence'}\n\n"
            "YOUR OUTPUT: One line, English PubMed syntax with AND/OR. Example:\n"
            '("chronic musculoskeletal pain") AND ("psychological intervention" OR "stress management")'
        )

        base = os.getenv("LLM_BASE_URL", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        if not base or not model:
            raise RuntimeError("Falta LLM_BASE_URL o LLM_MODEL para LlmQueryPlanner")

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
        temp = min(float(os.getenv("LLM_TEMPERATURE", "0.1")), self._temp_cap)

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
        if not raw.strip():
            raise RuntimeError("LLM devolvió respuesta vacía")

        line0 = pubmed_line_from_llm_text(raw)
        if is_spanish_pubmed_line(line0):
            retry_body = (
                "IMPORTANT: Your previous answer was in Spanish. You MUST write in ENGLISH only.\n\n"
                "Translate this Spanish clinical question into an English PubMed search string:\n"
                f"{q_clip or '(none)'}\n\n"
                "Write ONE line in English PubMed syntax (AND/OR). Example:\n"
                '("chronic pain" OR "musculoskeletal pain") AND ("psychological intervention" OR "CBT")'
            )
            raw2 = _openai_chat_completion_text(
                base_url=base,
                api_key=api_key,
                model=model,
                system=_SYSTEM,
                user=retry_body,
                max_tokens=self._max_out,
                temperature=temp,
                timeout_s=self._timeout,
            )
            if not raw2.strip():
                raise RuntimeError("LLM retry vacío")
            line0 = pubmed_line_from_llm_text(raw2)

        out = finalize_llm_pubmed_line(line0)
        if not out.strip():
            raise RuntimeError("Query PubMed vacía tras post-proceso")
        return out
