"""
Perfil LLM desde ``.env``: elegir OpenAI (cloud), llama.cpp (OpenAI-compatible ``/v1``)
o desactivar el uso de LLM en el planificador PubMed.

``LlmQueryPlanner`` solo habla **OpenAI-compatible** ``POST .../chat/completions``; el servidor
llama.cpp debe exponer la API tipo OpenAI (p. ej. ``llama-server`` con ``/v1``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def _first_nonempty_str(data: Mapping[str, Any | None], keys: tuple[str, ...]) -> str:
    for k in keys:
        raw = data.get(k)
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            return s
    return ""


def apply_copilot_llm_profile_from_dotenv(path: Path) -> str:
    """
    Lee ``COPILOT_LLM_PROFILE`` del fichero ``.env`` (vía ``dotenv_values``) y aplica
    ``LLM_BASE_URL``, ``LLM_MODEL`` y ``OPENAI_API_KEY`` en ``os.environ``.

    Valores de perfil (minúsculas / alias):

    - ``custom`` (default): no modifica LLM; usa ``LLM_*`` ya cargados.
    - ``off`` / ``none`` / ``disabled``: fuerza ``COPILOT_QUERY_PLANNER=heuristic``.
    - ``openai`` / ``openai_cloud`` / ``chatgpt``: toma ``OPENAI_LLM_*`` / ``OPENAI_API_KEY``.
    - ``llamacpp`` / ``llama_cpp`` / ``ollama``: toma ``LLAMACPP_LLM_*`` y clave opcional.

    Returns:
        Nombre canónico del perfil aplicado (``custom``, ``off``, ``openai``, ``llamacpp``).
    """
    try:
        from dotenv import dotenv_values  # pyright: ignore[reportMissingImports]
    except ImportError:
        return "custom"

    if not path.is_file():
        os.environ.setdefault("COPILOT_LLM_PROFILE", "custom")
        return "custom"

    data = dotenv_values(path)
    raw = (data.get("COPILOT_LLM_PROFILE") or os.getenv("COPILOT_LLM_PROFILE") or "custom").strip().lower()
    if not raw or raw in ("custom", "manual", "default"):
        os.environ["COPILOT_LLM_PROFILE"] = "custom"
        return "custom"

    if raw in ("off", "none", "disabled", "no"):
        os.environ["COPILOT_QUERY_PLANNER"] = "heuristic"
        os.environ["COPILOT_LLM_PROFILE"] = "off"
        return "off"

    if raw in ("openai", "openai_cloud", "chatgpt"):
        base = _first_nonempty_str(
            data,
            ("OPENAI_LLM_BASE_URL", "OPENAI_API_BASE", "OPENAI_BASE_URL", "LLM_BASE_URL"),
        )
        model = _first_nonempty_str(
            data,
            ("OPENAI_LLM_MODEL", "OPENAI_MODEL", "OPENAI_CHAT_MODEL", "LLM_MODEL"),
        )
        key = _first_nonempty_str(data, ("OPENAI_API_KEY", "OPENAI_LLM_API_KEY"))
        if base:
            os.environ["LLM_BASE_URL"] = base
        if model:
            os.environ["LLM_MODEL"] = model
        if key:
            os.environ["OPENAI_API_KEY"] = key
        os.environ["COPILOT_LLM_PROFILE"] = "openai"
        return "openai"

    if raw in ("llamacpp", "llama_cpp", "ollama", "local"):
        base = _first_nonempty_str(
            data,
            ("LLAMACPP_LLM_BASE_URL", "LLAMA_CPP_OPENAI_BASE_URL", "LLM_BASE_URL"),
        )
        model = _first_nonempty_str(
            data,
            ("LLAMACPP_LLM_MODEL", "LLAMA_CPP_MODEL", "LLM_MODEL"),
        )
        key = _first_nonempty_str(
            data,
            ("LLAMACPP_OPENAI_API_KEY", "LLAMA_CPP_API_KEY", "OPENAI_API_KEY"),
        )
        if base:
            os.environ["LLM_BASE_URL"] = base.rstrip("/")
        if model:
            os.environ["LLM_MODEL"] = model
        if key:
            os.environ["OPENAI_API_KEY"] = key
        else:
            # Muchos servidores locales ignoran la clave; OpenAI exige header no vacío a veces.
            os.environ.setdefault("OPENAI_API_KEY", "sk-local-llamacpp")
        os.environ["COPILOT_LLM_PROFILE"] = "llamacpp"
        return "llamacpp"

    os.environ["COPILOT_LLM_PROFILE"] = raw
    return raw
